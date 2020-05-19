#-*- coding:utf-8 -*-
# author: Racle
# project: pointer-network

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import random

import config


use_cuda = config.use_gpu and torch.cuda.is_available()


random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class Model:
    def __init__(self, model_file_path=None, is_eval=False):
        "model_file_path: 从头训练时设置为None，保存模型在log文件下的对应时间的train dir下。"
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()
        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])


class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        # self.embedding.from_pretrained(weights)
        nn.init.normal_(self.embedding.weight, 0, config.norm_init_std)

        self.lstm = nn.LSTM(config.emb_dim,
                            config.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0)  # 可调节
        # 特征输出线性变换：bidirectional=True，乘以2
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, input, seq_lens):
        """

        :param input:
        :param seq_lens:
        :return:
            encoder_outputs: 计算context vector，原始的lstm每个时刻的输出。
            encoder_feature: Wh * hi 进行attention weight计算。
            hidden： decoder initial state
        """
        embedded = self.embedding(input)

        # seq_lens should be in descending order. pack之后，rnn会直接忽略pad的输出，只输出不是pad的输出。
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        # pad_packed_sequence: inverse operation to pack_padded_sequence。转换为正常的数据结构。
        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # B x e_i x 2*hidden_dim
        encoder_outputs = encoder_outputs.contiguous()  # 新开辟内存，储存新值为按行储存的tensor
                                                        # 在view前，考虑是否在内存上修改储存方式
        encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim)  # B * e_i x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden


class ReduceState(nn.Module):
    def __init__(self,):
        """bidirectional的last hidden state和last cell state处理，压缩成一个hidden size维度。作为decoder的初始化。"""
        super(ReduceState, self).__init__()
        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        nn.init.normal_(self.reduce_h.weight, 0, config.norm_init_std)
        nn.init.zeros_(self.reduce_h.bias)

        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        nn.init.normal_(self.reduce_c.weight)
        nn.init.zeros_(self.reduce_c.bias)

    def forward(self, hidden):
        h, c = hidden  # 2 x B x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))
        # 1 x B x hidden_dim
        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention的累积分布加入attention weight的计算，防止重复生成相同的词或短语。同时在计算损失时，加入coverage loss。
        if config.do_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        # 线性变化
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        """decoder运行到t时刻时，计算此时的attention weight。

        :param s_t_hat: 来自decoder，输入计算attention weight。concat了decoder的h和c。维度：B x 2*hidden_dim
        :param encoder_outputs: 计算context vector。
        :param encoder_feature: Wh * hi 进行attention weight计算的部分
        :param enc_padding_mask: 记录输入文本的padding的mask。
        :param coverage: 前t-1步累计attention dist
        :return:
            context_v, context vector
            attn_dist, decoder运行到t时刻的attention分布
            coverage, attention的累积分布
        """
        # e_i: 需要summary的文本的tokens的长度。文章中，encoder中i为e_i维度上的index
        b, e_i, n = list(encoder_outputs.size())  # n = 2*hidden_dim

        # decoder输出，计算attention weight的数据
        dec_fea = self.decode_proj(s_t_hat)                                    # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, e_i, n).contiguous() # B x e_i x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)                        # B*e_i x 2*hidden_dim
        att_features = encoder_feature + dec_fea_expanded                      # B*e_i x 2*hidden_dim

        if config.do_coverage:
            coverage_input = coverage.view(-1, 1)                              # B*e_i x 1
            coverage_feature = self.W_c(coverage_input)                        # B*e_i x 2*hidden_dim
            att_features = att_features + coverage_feature

        # attention weight
        e = F.tanh(att_features)                                               # B * e_i x 2*hidden_dim
        scores = self.v(e)                                                     # B * e_i x 1
        scores = scores.view(-1, e_i)                                          # B x e_i

        # padding部分处理
        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask               # B x e_i
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor                          # B x e_i

        # 计算context vector
        attn_dist = attn_dist.unsqueeze(1)                                     # B x 1 x e_i
        # batch matrix-matrix，does not broadcast.
        # [B x 1 x e_i] bmm [B x e_i x 2*hidden_dim]
        context_v = torch.bmm(attn_dist, encoder_outputs)                      # B x 1 x 2*hidden_dim
        context_v = context_v.view(-1, config.hidden_dim * 2)                  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, e_i)                                    # B x e_i

        # 更新coverage
        if config.do_coverage:
            coverage = coverage.view(-1, e_i)
            coverage = coverage + attn_dist

        return context_v, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()

        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        # nn.init.normal_(self.embedding.weight, 0, config.norm_init_std)  # embedding共享
        # 输入为: encoder的context vector + t时刻输入的word embedding
        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim,
                            config.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        # 输入：context vector，decoder的h和c，以及t时刻的decoder的输入x，计算p_gen。
        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # 每一步计算词表上的预测分布
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        nn.init.normal_(self.out1.weight)
        nn.init.zeros_(self.out1.bias)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        nn.init.normal_(self.out2.weight)
        nn.init.zeros_(self.out2.bias)

    def forward(self, d_inp, d_hc, encoder_outputs, encoder_feature, enc_padding_mask,
                init_context_v, extra_zeros, enc_batch_extend_vocab, coverage, step):
        """

        :param d_inp: decoder的word embedding输入
        :param d_hc: decoder的h和c
        :param encoder_outputs: 见Attention
        :param encoder_feature: 见Attention
        :param enc_padding_mask: 见Attention
        :param init_context_v: 第一个时刻，的context vector, 默认为 0
        :param extra_zeros: 作为OOV词汇的占位符存在，以保留文章中OOV单词的概率
        :param enc_batch_extend_vocab: the enc_input where in-article OOVs are represented by their temporary OOV id.
        :param coverage: 见Attention
        :param step: 训练步数
        :return:
        """
        # 测试时初始步骤
        if not self.training and step == 0:
            h_decoder, c_decoder = d_hc
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            context_v_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                              enc_padding_mask, coverage)
            coverage = coverage_next

        # decoder step
        d_inp_embd = self.embedding(d_inp)
        x = self.x_context(torch.cat((init_context_v, d_inp_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), d_hc)

        # 计算 attention 部分
        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        context_v_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                          enc_padding_mask, coverage)
        if self.training or step > 0:
            coverage = coverage_next

        # 计算使用生成的word的概率。
        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((context_v_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        # 计算generator的输出词分布
        output = torch.cat((lstm_out.view(-1, config.hidden_dim), context_v_t), 1)  # B x hidden_dim * 3
        output = self.out1(output)                                                  # B x hidden_dim
        output = nn.Dropout(0.2)(output)
        output = self.out2(output)                                                  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        # 计算加权的最终分布
        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist
            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)              # B x (vocab_size+extra_zeros num)
            # scatter_add: 按照enc_batch_extend_vocab给出的index，将attention distribution value加到vocab distribution中。
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, context_v_t, attn_dist, p_gen, coverage