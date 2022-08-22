import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """

        :param d_model:
        :param vocab: 词汇量
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x)*math.sqrt(self.d_model)



if __name__ == '__main__':
    d_model = 512
    vocab = 1000

    x = Variable(torch.LongTensor([[100, 2, 421, 50], [491, 998, 2, 221]]))

    emb = Embeddings(d_model, vocab)
    embr = emb(x)
    print("embr:", embr, embr.max(), embr.min())
    print(emb)
    print(embr.shape)
    print(math.sqrt(d_model))

