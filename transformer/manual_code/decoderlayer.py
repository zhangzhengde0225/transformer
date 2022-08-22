import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import positionalencoding
import embedding
import mutiheadattention
import PositionwiseFeedForward as pff
import encoderlayer


class Deconderlayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(Deconderlayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = mutiheadattention.clones(pff.SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m , m, source_mask))
        return self. sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, N, self_attn, src_attn,ff, size, dropout):
        super(Decoder, self).__init__()
        self.layers = []
        for i in range(N):
            layer = Deconderlayer(size=size, self_attn=self_attn, src_attn = src_attn,feed_forward=ff, dropout=dropout)
            self.layers.append(layer)
        features = self.layers[0].size  # int 512
        self.norm = pff.LayerNorm(features)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


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


    self_attn = src_attn = mutiheadattention.MultiHeadAttention(head, d_model)
    ff = pff.PositionwiseFeedForward(d_model, d_ff, dropout)
    mask = Variable(torch.zeros(8, 4, 4))
    source_mask = target_mask = mask

    el = encoderlayer.EncoderLayer(size, self_attn, ff, dropout)
    el_result = el(x, mask)
    print(el_result)
    print(el_result.shape)

    # c = copy.deepcopy
    # layer = EncoderLayer(size, c(self_attn), c(ff), dropout)

    N = 8
    en = encoderlayer.Encoder(N, self_attn, ff, size, dropout)
    en_result = en(x, mask)
    print('en_result', en_result, en_result.shape)

    memory = en_result

    dl = Deconderlayer(size, self_attn, src_attn, ff, dropout)
    dl_result = dl(x, memory, source_mask, target_mask)
    print('dl_reslult:', dl_result, dl_result.shape)

    # layer = Deconderlayer(d_model, self_attn, src_attn, ff, dropout)

    de = Decoder(N, self_attn, src_attn,ff, size, dropout)
    de_result = de(x, memory, source_mask, target_mask)
    print('de_result:', de_result, de_result.shape)

    x = de_result
    gen = Generator(d_model, vocab)
    gen_result = gen(x)
    print('gen_result:', gen_result, gen_result.shape)




