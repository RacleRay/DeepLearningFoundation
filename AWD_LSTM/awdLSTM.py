import torch
import torch.functional as F
import torch.jit as jit
from torch import nn
from torch import Tensor


class LSTMCell(jit.ScriptModule):
    def __init__(self, ni, nh):
        super().__init__()
        self.ni = ni
        self.nh = nh
        self.Wih = nn.Parameter(torch.randn(4 * nh, ni))
        self.Whh = nn.Parameter(torch.randn(4 * nh, nh))
        self.bias_ih = nn.Parameter(torch.randn(4 * nh))
        self.bias_hh = nn.Parameter(torch.randn(4 * nh))

    @jit.script_method
    def forward(self, input:Tensor, state:Tuple[Tensor,Tensor])->Tuple[Tensor, Tuple[Tensor, Tensor]]:
        h, c = state
        gates = (input @ self.Wih + self.bias_ih + h @ self.Whh + self.bias_hh)
        inGate, forgetGate, cellActivate, outGate = gates.chuck(4, 1)

        inGate = torch.sigmoid(inGate)
        forgetGate = torch.sigmoid(forgetGate)
        cellActivate = torch.tanh(cellActivate)
        outGate = torch.sigmoid(outGate)

        cout = (forgetGate * c) + (inGate * cellActivate)
        hout = outGate * torch.tanh(cout)

        return hout, (hout, cout)


class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input:Tensor, state:Tuple[Tensor, Tensor])->Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(1)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs.append(out)
        return torch.stack(outputs, dim=1), state


# We'll need all different kinds of dropouts.
# Dropout consists into replacing some coefficients by 0 with probability p.
# To ensure that the average of the weights remains constant,
# we apply a correction to the weights that aren't nullified of a factor 1/(1-p).

def dropoutMask(x, size, prob):
    "prob -- replaced by 0 with this probability"
    return x.new(*size).bernoulli_(1-prob).div_(1-prob)

# A tensor x will have three dimensions: bs, seq_len, vocab_size.
# We consistently apply the dropout mask across the seq_len dimension
class RNNDropout(nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0:
            return x
        mask = dropoutMask(x.data, (x.size(0), 1, x.size(2)), self.prob)
        return x * mask

WEIGHT_HH = 'weight_hh_l0'

class WeightDropout(nn.Module):
    "WeightDropout is dropout applied to the weights of the inner LSTM hidden to hidden matrix."
    def __init__(self, module, weight_prob=[0.], layer_names=[WEIGHT_HH]):
        super().__init__()
        self.module = module
        self.weight_prob = weight_prob
        self.layer_names = layer_names
        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_prob, training=False)

    def _setWeights(self):
        """if we want to preserve the CuDNN speed and not reimplement the cell from scratch.
        We add a parameter that will contain the raw weights,
        and we replace the weight matrix in the LSTM at the beginning of the forward pass."""
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_pro, training=self.training)

    def forward(self, *args):
        import warnings
        self._setWeights()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.module.forward(*args)


class EmbeddingDropout(nn.Module):
    "EmbeddingDropout applies dropout to full rows of the embedding matrix."
    def __init__(self, embedding, embed_prob):
        super().__init__()
        self.embedding = embedding
        self.embed_prob = embed_prob
        self.pad_idx = self.embedding.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words, scale=None):
        if self.training and self.embed_p != 0:
            size = (self.embedding.weight.size(0), 1)
            mask = dropoutMask(self.embedding.weight.data, size, self.embed_prob)
            maskedEmbedding = self.embedding.weight * mask
        else:
            maskedEmbedding = self.embedding.weight
        if scale:
            maskedEmbedding.mul_(scale)
        return F.embedding(words, maskedEmbedding, self.pad_idx,
                        self.embedding.max_norm, self.embedding.normtype,
                        self.embedding.scale_grad_by_freq, self.embedding.sparse)


def to_detach(h):
    return h.detach() if type(h) == torch.Tensor else tuple(to_detach(v) for v in h)

class AWDLSTM(nn.Module):
    initRange = 0.1
    "AWD-LSTM: https://arxiv.org/abs/1708.02182."
    def __init__(self, vocab_size, embed_size, n_hid, n_layers, pad_token,
                hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5):
        super().__init__()
        self.batch_size = 1
        self.embed_size = embed_size
        self.n_hid = n_hid
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token)
        self.embedding_dropout = EmbeddingDropout(self.embedding, embed_p)
        self.rnns = [nn.LSTM(embed_size if l==0 else n_hid,
                             (n_hid if l != n_layers - 1 else embed_size),
                             1, batch_first=True) for l in range(n_layers)]
        self.rnns = nn.ModuleList([WeightDropout(rnn, weight_p) for rnn in self.rnns])
        self.embedding.weight.data.uniform_(-self.initRange, self.initRange)
        self.input_dropout = RNNDropout(input_p)
        self.hidden_dropout = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def _init_hidden(self, l):
        "Return one hidden state."
        nh = self.n_hid if l != self.n_layers - 1 else self.embed_size
        return next(self.parameters()).new(1, self.batch_size, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        self.hidden = [(self._init_hidden(l), self._init_hidden(l)) for l in range(self.n_layers)]

    def forward(self, input):
        batch_size, seq_len = input.size()
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.reset()
        raw_output = self.input_dropout(self.embedding_dropout(input))
        new_hidden, raw_outputs, outputs = [], [], []
        for l, (rnn, hidden_dropout) in enumerate(zip(self.rnns, self.hidden_dropout)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1:
                raw_output = hidden_dropout(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden)
        return raw_outputs, outputs


