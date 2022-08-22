import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

import positionalencoding
import embedding
import copy


def subsequent_mask(bs, size):
    attn_shape = (bs, size, size)  # 元组(1,5,5)
    mask_ = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(1 - mask_)


def attention(query, key, value, mask=None, dropout=None):
    """

    :param query: 文本
    :param key: 键
    :param value: 值
    :param mask: 掩码张量
    :param dropout: 随机丢弃
    :return: tensor
    """

    # d_k = query.size(-1)  # query最后一维，词嵌入维度
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)  # query和key的转置进行矩阵乘法，除以缩放系数
    # (2,4,4)
    if mask is not None:  # 判断是否使用掩码张量
        scores = scores.masked_fill(mask == 0, -1e9)  # 把mask为0的位置的score值替换为-1e9
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:  # 判断是否使用dropout
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # 完成张量乘法，返回注意力表示


def clones(module, N):
    x = [copy.deepcopy(module) for _ in range(N)]  # 列表，长度4, 元素是fc, [fc, fc, fc, fc]
    x = nn.ModuleList(x)  # 克隆N个独立的层
    return x


class MultiHeadAttention(nn.Module):

    def __init__(self, head, embedding_dim, dropout=0.1):
        """

        :param head: 头数, 4
        :param embedding_dim: 词嵌入维度 512 int
        :param dropout: 置0比率
        """
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % head == 0  # 判断h能否被d_model整除，给每个头分配等量的词特征
        self.d_k = embedding_dim // head  # 每个头获得的词向量维度
        self.head = head
        fc = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.linears = clones(fc, 4)  # nn.ModuleList [fc, fc ,fc ,fc]
        # 线性层内部变换矩阵embedding_dim * embedding_dim，Q,K,V还有最后拼接的矩阵各需要一个linear
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """

        :param query: (2,4,512)
        :param key:
        :param value:
        :param mask: (2,4,4)
        :return: (2, 4, 512)
        """
        if mask is not None:  # 掩码张量
            mask = mask.unsqueeze(1)  # (2,1,4,4)
        batch_size = query.size(0)  # query的第一维，代表多少条样本 2
        # query, key, value = \
        #     [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)  # (2,4,4,128)
        #      for model, x in zip(self.linears, (query, key, value))]
        tmp = []
        for model, x in zip(self.linears, (query, key, value)):
            a = model(x)
            b = a.reshape(batch_size, -1, self.head, self.d_k)
            c = b.transpose(1, 2)
            tmp.append(c)

        query = tmp[0]  # (2,4,4,128)
        key = tmp[1]    # (2,4,4,128)
        value = tmp[2]  # (2,4,4,128)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)  # x:(2,4,4,128) self.attn (2,4,4,4)
        x = x.transpose(1,2).contiguous().view(batch_size, -1, self.head * self.d_k)  # (2,4,512)
        return self.linears[-1](x)


if __name__ == '__main__':
    # size = 5
    # mask = subsequent_mask(1, size)
    # print(mask)

    # plt.figure(figsize=(5, 5))
    # sm = subsequent_mask(20)[0]  # (20,20)
    # plt.plot(sm)
    # plt.imshow(sm)
    # plt.show()

    d_model = 512
    dropout = 0.2
    vocab = 1000
    x = Variable(torch.LongTensor([[100, 2, 421, 50], [491, 998, 2, 221]]))

    emb = embedding.Embeddings(d_model, vocab)
    embr = emb(x)  # (2,4,512)
    poen = positionalencoding.PositionalEncoding(d_model, dropout, max_len=5000)
    ppp = poen(x=embr)
    print(f'ppp: {ppp}')

    query = key = value = ppp
    mask = Variable(torch.zeros(2, 4, 4))
    # mask = subsequent_mask(2, 4)
    print(mask)
    attn, p_attn = attention(query, key, value, mask=mask, dropout=None)
    print('attn:', attn)
    print(attn.shape)
    print('p_attn:', p_attn)
    print(p_attn.shape)

    head = 8
    embedding_dim = 512
    mask = Variable(torch.zeros(8, 4, 4))
    mha = MultiHeadAttention(head, embedding_dim, dropout)
    mha_result = mha(query,key,value,mask)
    print(mha_result)
    print(mha_result.shape)



