
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
from collections import OrderedDict

random.seed(1024)

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

# FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
# ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


class RNTN(nn.Module):
    "https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf"

    def __init__(self, word2index, hidden_size, output_size):
        super(RNTN, self).__init__()

        self.word2index = word2index
        self.embed = nn.Embedding(len(word2index), hidden_size)

        self.V = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size * 2, hidden_size * 2))
            for _ in range(hidden_size)
        ])  # Tensor: hidden_size * 2hidden_size * 2hidden_size
        self.W = nn.Parameter(torch.randn(hidden_size * 2, hidden_size))
        self.b = nn.Parameter(torch.randn(1, hidden_size))

        self.W_out = nn.Linear(hidden_size, output_size)

    def init_weight(self):
        nn.init.xavier_uniform(self.embed.state_dict()['weight'])
        nn.init.xavier_uniform(self.W_out.state_dict()['weight'])
        for param in self.V.parameters():
            nn.init.xavier_uniform(param)
        nn.init.xavier_uniform(self.W)
        self.b.data.fill_(0)

    def tree_propagation(self, node):
        recursive_tensor = OrderedDict()
        current = None
        if node.isLeaf:
            tensor = Variable(LongTensor([self.word2index[node.word]])) if node.word in self.word2index.keys() \
                          else Variable(LongTensor([self.word2index['<UNK>']]))
            current = self.embed(tensor)  # 1x hidden_size
        else:
            recursive_tensor.update(self.tree_propagation(node.left))
            recursive_tensor.update(self.tree_propagation(node.right))

            concated = torch.cat(
                [recursive_tensor[node.left], recursive_tensor[node.right]],
                1)  # 1x 2hidden_size
            xVx = []
            for i, v in enumerate(self.V):
                xVx.append(
                    torch.matmul(torch.matmul(concated, v),
                                 concated.transpose(0, 1)))
            xVx = torch.cat(xVx, 1)
            Wx = torch.matmul(concated, self.W)  # 1x hidden_size

            current = F.tanh(xVx + Wx + self.b)  # 1x hidden_size
        recursive_tensor[node] = current
        return recursive_tensor

    def forward(self, Trees, root_only=False):
        "Trees: Stanford Sentiment Treebank."
        propagated = []
        if not isinstance(Trees, list):
            Trees = [Trees]
        for Tree in Trees:
            recursive_tensor = self.tree_propagation(Tree.root)
            if root_only:
                recursive_tensor = recursive_tensor[Tree.root]
                propagated.append(recursive_tensor)
            else:
                recursive_tensor = [
                    tensor for node, tensor in recursive_tensor.items()
                ]
                propagated.extend(recursive_tensor)

        propagated = torch.cat(
            propagated)  # (num_of_node in batch, hidden_size)

        return F.log_softmax(self.W_out(propagated), 1)
