import torch
import torch.nn as nn
from torch.autograd import Variable

import positionalencoding
import embedding
import mutiheadattention
import PositionwiseFeedForward as pff


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = mutiheadattention.clones(pff.SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        mha_func = lambda x: self.self_attn(x, x, x, mask)
        x = self.sublayer[0](x, mha_func)
        ff_func = self.feed_forward
        x = self.sublayer[1](x, ff_func)
        return x


class Encoder(nn.Module):
    def __init__(self, N, attn, ff, size, dropout):
        super(Encoder, self).__init__()
        # self.layers = mutiheadattention.clones(layer, N)  # nn.Modulelist [] N个元素

        # self.layers = [EncoderLayer(size=size, self_attn=attn, feed_forward=ff, dropout=dropout) for _ in range(N)]
        self.layers = []
        for i in range(N):
            layer = EncoderLayer(size=size, self_attn=attn, feed_forward=ff, dropout=dropout)
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)
        # print('xx', self.layers[0].size)
        features = self.layers[0].size  # int 512
        self.norm = pff.LayerNorm(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


if __name__ == '__main__':
    size = d_model = 512
    head = 8
    d_ff = 64
    vocab = 1000
    dropout = 0.2
    x = Variable(torch.LongTensor([[100, 2, 421, 50], [491, 998, 2, 221]]))

    emb = embedding.Embeddings(d_model, vocab)
    embr = emb(x)  # (2,4,512)
    poen = positionalencoding.PositionalEncoding(d_model, dropout, max_len=5000)
    ppp = poen(x=embr)
    x = ppp

    self_attn = mutiheadattention.MultiHeadAttention(head, d_model)
    ff = pff.PositionwiseFeedForward(d_model, d_ff, dropout)
    mask = Variable(torch.zeros(8, 4, 4))

    el = EncoderLayer(size, self_attn, ff, dropout)
    el_result = el(x, mask)
    print(el_result)
    print(el_result.shape)

    # c = copy.deepcopy
    # layer = EncoderLayer(size, c(self_attn), c(ff), dropout)

    N = 8
    en = Encoder(N, self_attn, ff, size, dropout)
    en_result = en(x, mask)
    print('en_result', en_result, en_result.shape)

