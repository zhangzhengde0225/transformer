import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import embedding



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """

        :param d_model: model的维度
        :param dropout: 随机丢弃的比例
        :param max_len: 句子最大词长
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # 位置编码矩阵 (5000, 512)
        position = torch.arange(0, max_len).unsqueeze(1)  # 绝对位置矩阵，(5000, 1)
        # print(position, position.shape)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))  # 取值范围(1, 0.0001)(256,)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列，
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        # pe: 取值范围（-1，1）
        pe = pe.unsqueeze(0)  # (1,5000,512)
        self.register_buffer('pe', pe)  # 注册缓存数据，ie新增一个叫pe的属性

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1), :], requires_grad=False)
        return self.dropout(x)



if __name__ == '__main__':
    d_model = 512
    dropout = 0.1
    vocab = 1000
    x = Variable(torch.LongTensor([[100, 2, 421, 50], [491, 998, 2, 221]]))

    emb = embedding.Embeddings(d_model, vocab)
    embr = emb(x)  # (2,4,512)
    poen = PositionalEncoding(d_model, dropout, max_len=5000)
    ppp = poen(x=embr)
    print(f'ppp: {ppp} {ppp.shape}')  # (2, 4 , 512)



