import torch
import torch.nn as nn
import numpy as np
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
from pyitcast.transformer_utils import greedy_decode


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    """

    :param source_vocab: 词表大小 11
    :param target_vocab: 词表大小 11
    :param N: 6 模块复制N次
    :param d_model: 嵌入维度 512
    :param d_ff: feed forward的维度
    :param head: 头数 8
    :param dropout: 随机丢弃比例
    :return: 返回整个transformer模型
    """
    attn = mutiheadattention.MultiHeadAttention(head, d_model)
    self_attn = mutiheadattention.MultiHeadAttention(head, d_model)
    src_attn = mutiheadattention.MultiHeadAttention(head, d_model)
    ff = pff.PositionwiseFeedForward(d_model, d_ff, dropout)
    position = positionalencoding.PositionalEncoding(d_model, dropout)

    a = encoderlayer.Encoder(N, attn, ff, d_model, dropout)  # 编码部分N倍
    b = decoderlayer.Decoder(N, self_attn, src_attn, ff, d_model, dropout)  # 解码部分N倍
    c = nn.Sequential(embedding.Embeddings(d_model, source_vocab), position)  # 输入部分
    d = nn.Sequential(embedding.Embeddings(d_model, target_vocab), position)  # 输出的输入部分
    e = decoderlayer.Generator(d_model, target_vocab)  # 概率输出部分
    model = encoderdecoder.EncoderDecoder(a, b, c, d, e)
    for p in model.parameters():  # 初始化参数
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def data_generator(V, batch_size, num_batch):
    """
    :param V: 生成数据的取值范围在[1, V]之间，V是词表量,11
    :param batch_size: bs 20
    :param num_batch: 30
    :return:
    """
    for i in range(num_batch):  # 30次
        data = np.random.randint(1, V, size=(batch_size, 10))  # (20, 10) 取值范围1~11的200个值
        data = torch.from_numpy(data)  # ()
        data[:, 0] = 1  # 所有行第0列置为1，
        source = Variable(data, requires_grad=False)  # (20, 10)
        target = Variable(data, requires_grad=False)  # (20, 10)

        # return Batch(source, target)
        yield Batch(source, target)  # Batch.__init__(source, target)


def run(model, loss, epochs=10, V=11, batch_size=8, num_batch=20):
    """

    :param model:
    :param loss:
    :param epochs: 10
    :param V: 11
    :param batch_size: 20
    :param num_batch: 30
    :return:
    """
    for epoch in range(epochs):  # 训练10代
        model.train()  # 解冻w和b
        data_loader = data_generator(V, batch_size=batch_size, num_batch=num_batch)
        mean_loss = run_epoch(data_loader, model, loss)  # 迭代30次，优化模型
        model.eval()  # 冻结w和b，准备评估模型
        data_loader2 = data_generator(V, batch_size=batch_size, num_batch=num_batch)
        run_epoch(data_loader2, model, loss)  # 迭代30次，不优化模型
        # accuracy = evaluate_model(data_loader2, model, loss)
        # print(accuracy)
    source = Variable(torch.LongTensor([[1, 3, 2, 5, 4, 6, 7, 8, 9, 10]]))
    source_mask = Variable(torch.ones(1, 1, 10))
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


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
    mask = Variable(torch.zeros(1, 4, 4))
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
    source_mask = target_mask = Variable(torch.zeros(1, 4, 4))
    ed = encoderdecoder.EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
    ed_result = ed(source, target, source_mask, target_mask)
    print('ed_result:', ed_result, ed_result.shape)

    source_vocab = 11  # 词汇量
    target_vocab = 11  # 词汇量
    N = 6  # layer层数
    res = make_model(source_vocab, target_vocab, N)
    print(res)

    V = 11  # 词汇量
    batch_size = 20
    num_batch = 30

    model = make_model(V, V, N=2)  # transformer模型
    model_optimizer = get_std_opt(model)  # adam优化器
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)  # nn.KLDivLoss
    loss = SimpleLossCompute(model.generator, criterion, model_optimizer)  # 计算损失函数的对象

    crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.5)  # nn.KLDivLoss
    # 5代表词汇量，01234
    predict = Variable(torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0]]))
    # 预测值是[2, 2, 2]
    target = Variable(torch.LongTensor([2, 1, 0]))
    crit(predict, target)
    # plt.imshow(crit.true_dist)
    # plt.show()

    epochs = 10
    print(f'model: {model}')
    run(model, loss, epochs=epochs, V=V, batch_size=batch_size, num_batch=num_batch)




