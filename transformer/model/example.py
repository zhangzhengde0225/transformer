import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data.utils import get_tokenizer
from pyitcast.transformer import TransformerModel

TEXT = torchtext.data.Field(tokenize=get_tokenizer('basic_english'), init_token='<sos>', eos_token='<eos>', lower=True)
print(TEXT)
train_text, val_text, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
print(test_txt.exaples[0].text[:10])
