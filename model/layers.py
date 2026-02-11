import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math

class positionWiseFeedForward(nn.Module):
  def __init__(self, embed_dim, dff, dropout=0.1):
    super(positionWiseFeedForward, self).__init__()

    self.fc1 = nn.Linear(embed_dim, dff)
    self.fc2 = nn.Linear(dff, embed_dim)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.fc2(self.relu(self.fc1(x)))


class positionalEncoding(nn.Module):
  def __init__(self,embed_dim,max_seq_length):
    super(positionalEncoding, self).__init__()

    pe = torch.zeros(max_seq_length,embed_dim)
    position = torch.arange(0 , max_seq_length, dtype= torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[: ,1::2] = torch.cos(position * div_term)

    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    return x + self.pe[: , : x.size(1)]


from torch.nn.modules import LayerNorm
class EncoderLayer(nn.Module):
  def __init__(self, embed_dim, heads, dff, dropout):
     super(EncoderLayer, self).__init__()

     self.self_attn = MultiHeadAttention(embed_dim, heads)
     self.ffn = positionWiseFeedForward(embed_dim, dff)
     self.norm1 = LayerNorm(embed_dim)
     self.norm2 = LayerNorm(embed_dim)
     self.dropout= nn.Dropout(dropout)

  def forward(self, x, mask):
    attn_output = self.self_attn(x, x, x, mask)
    x = self.norm1(x+ self.dropout(attn_output))

    ff= self.ffn(x)
    x= self.norm2(x + self.dropout(ff))

    return x


