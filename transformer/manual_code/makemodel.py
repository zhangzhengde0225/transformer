import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

import positionalencoding
import embedding
import mutiheadattention
import PositionwiseFeedForward as pff
import encoderlayer
import decoderlayer
import encoderdecoder

from pyitcast.transformer_utils import get_std_opt
from pyitcast.transformer_utils import LabelSmoothing
from pyitcast.transformer_utils import SimpleLossCompute
from pyitcast.transformer_utils import run_epoch
from pyitcast.transformer_utils import Batch


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    attn = mutiheadattention.MultiHeadAttention(head, d_model)
    ff = pff.PositionwiseFeedForward(d_model, d_ff, dropout)
    position = positionalencoding.PositionalEncoding(d_model, dropout)
    a = encoderlayer.Encoder(N, attn, ff, d_model, dropout)
    b = decoderlayer.Decoder(N, self_attn, src_attn, ff, d_model, dropout)
    c = nn.Sequential(embedding.Embeddings(d_model, source_vocab), position)
    d = nn.Sequential(embedding.Embeddings(d_model, target_vocab), position)
    e = decoderlayer.Generator(d_model, target_vocab)
    model = encoderdecoder.EncoderDecoder(a, b, c, d, e)
    # model = encoderdecoder.EncoderDecoder(encoderlayer.Encoder(encoderlayer.EncoderLayer(d_model, attn, ff, dropout), N),
    #         decoderlayer.Decoder(decoderlayer.Deconderlayer(d_model, attn, attn, ff, dropout),N),
    #         nn.Sequential(embedding.Embeddings(d_model, source_vocab, position)),
    #         nn.Sequential(embedding.Embeddings(d_model, target_vocab, position)),
    #         decoderlayer.Generator(d_model, target_vocab))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def data_generator(V, batch_size, num_batch):
    """
    :param V: (1,V)取任意整数，11
    :param batch_size:
    :param num_batch:
    :return:
    """
    for i in  range(num_batch):
        data = torch.from_numpy((np.random.randint(1, V, size=(batch_size, 10))))
        data[:, 0] = 1
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        yield Batch(source, target)


def run(model, loss, epochs=10):
    for epoch in range(epochs):
        model.train()
        run_epoch(data_generator(V, 8, 20), model, loss)
        model.eval()
        run_epoch(data_generator(V, 8, 5), model, loss)


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
    # print(el_result)
    # print(el_result.shape)
    N = 8
    en = encoderlayer.Encoder(N, self_attn, ff, size, dropout)
    en_result = en(x, mask)
    # print('en_result', en_result, en_result.shape)

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
    source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    source_mask = target_mask = Variable(torch.zeros(8, 4, 4))
    ed = encoderdecoder.EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
    ed_result = ed(source, target, source_mask, target_mask)
    print('ed_result:', ed_result, ed_result.shape)

    source_vocab = 11
    target_vocab = 11
    N = 6
    res = make_model(source_vocab, target_vocab, N)
    print(res)


    V = 11
    batch_size =20
    num_batch = 30

    model = make_model(V, V, N=2)
    model_optimizer = get_std_opt(model)
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    loss = SimpleLossCompute(model.generator, criterion, model_optimizer)

    crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.5)
    predict = Variable(torch.FloatTensor([0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0]))
    target = Variable(torch.LongTensor[2, 1, 0])
    crit(predict, target)
    plt.imshow(crit.true_dist)
    plt.show()

    epochs = 10
    run(model, loss)




