import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    #  intilaztion of dimentions
    def __init__(self, embed_dim, heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k = embed_dim // heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    #  scaled dot product attention
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores= torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
           attn_scores= attn_scores.masked_fill(mask==0,-1e9)

        attn_probs= torch.softmax(attn_scores, dim=-1)
        output= torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, embed_dim = x.size()
    # reshape to [batch, seq_length, heads, d_k] → transpose to [batch, heads, seq_length, d_k]
        return x.view(batch_size, seq_length, self.heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
    # x: [batch, heads, seq_length, d_k] → [batch, seq_length, embed_dim]
        batch_size, heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
    def forward(self, Q, K, V, mask=None):
          Q = self.split_heads(self.W_q(Q))
          K = self.split_heads(self.W_k(K))
          V = self.split_heads(self.W_v(V))

          attn_output= self.scaled_dot_product_attention(Q, K, V, mask)

          output = self.W_o(self.combine_heads(attn_output))
          return output
