import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import positionalencoding
import embedding
import mutiheadattention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2*(x-mean)/(std+self.eps)+self.b2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


if __name__ == '__main__':
    d_model = 512
    dropout = 0.2
    vocab = 1000
    x = Variable(torch.LongTensor([[100, 2, 421, 50], [491, 998, 2, 221]]))

    emb = embedding.Embeddings(d_model, vocab)
    embr = emb(x)  # (2,4,512)
    poen = positionalencoding.PositionalEncoding(d_model, dropout, max_len=5000)
    ppp = poen(x=embr)
    # print(f'ppp: {ppp}')

    query = key = value = ppp
    mask = Variable(torch.zeros(2, 4, 4))
    # mask = subsequent_mask(2, 4)
    # print(mask)
    attn, p_attn = mutiheadattention.attention(query, key, value, mask=mask, dropout=None)
    # print('attn:', attn)
    # print(attn.shape)
    # print('p_attn:', p_attn)
    # print(p_attn.shape)

    head = 8
    embedding_dim = 512

    mha = mutiheadattention.MultiHeadAttention(head, embedding_dim, dropout)
    mha_result = mha(query, key, value, mask)
    print(mha_result)
    print(mha_result.shape)

    d_ff =64
    ff = PositionwiseFeedForward(d_model,d_ff,dropout)
    ff_result =ff(mha_result)
    print(ff_result)

    features = d_model = 512
    eps = 1e-6
    ln = LayerNorm(features, eps)
    ln_result = ln(ff_result)
    print(ln_result)

    size = 512
    x = ppp
    self_attn = mutiheadattention.MultiHeadAttention(head, d_model)
    sublayer = lambda x : self_attn(x, x, x, mask)

    sc = SublayerConnection(size, dropout)
    sc_result = sc(x, sublayer)
    print(sc_result)
    print(sc_result.shape)







