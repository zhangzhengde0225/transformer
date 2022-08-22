import torch
import torch.nn as nn
from torch.autograd import Variable

import positionalencoding
import embedding
import mutiheadattention
import PositionwiseFeedForward as pff
import encoderlayer
import decoderlayer


class EncoderDecoder(nn.Module):
    """组装好的transformer模型"""
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        encode_ret = self.encode(source, source_mask)
        return self.decode(encode_ret, source_mask, target, target_mask)

    def forward1(self, source, target, source_mask, target_mask):
        encode_ret = self.encode(source, source_mask)
        decode_ret = self.decode(encode_ret, source_mask, target, target_mask)
        return self.generator(decode_ret)

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        tgt_ret = self.tgt_embed(target)
        return self.decoder(tgt_ret, memory, source_mask, target_mask)


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
    N = 8
    en = encoderlayer.Encoder(N, self_attn, ff, size, dropout)
    en_result = en(x, mask)
    print('en_result', en_result, en_result.shape)

    memory = en_result

    dl = decoderlayer.Deconderlayer(size, self_attn, src_attn, ff, dropout)
    dl_result = dl(x, memory, source_mask, target_mask)
    print('dl_reslult:', dl_result, dl_result.shape)

    # layer = Deconderlayer(d_model, self_attn, src_attn, ff, dropout)

    de = decoderlayer.Decoder(N, self_attn, src_attn, ff, size, dropout)
    de_result = de(x, memory, source_mask, target_mask)
    print('de_result:', de_result, de_result.shape)

    x = de_result
    gen = decoderlayer.Generator(d_model, vocab)
    gen_result = gen(x)
    print('gen_result:', gen_result, gen_result.shape)
    encoder = en
    decoder = de
    source_embed = nn.Embedding(vocab, d_model)
    target_embed = nn.Embedding(vocab, d_model)
    generator = gen
    source = target =Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    source_mask = target_mask = Variable(torch.zeros(8, 4, 4))
    ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
    ed_result = ed(source, target, source_mask, target_mask)
    print('ed_result:', ed_result, ed_result.shape)
