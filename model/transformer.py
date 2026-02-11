import torch
import torch.nn as nn
from .layers import EncoderLayer, DecoderLayer, positionalEncoding

class Transformer (nn.Module):
  def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, heads, num_layers, dff, max_seq_length, dropout):
     super(Transformer, self).__init__()

     self.encoder_embedding = nn.Embedding(src_vocab_size, embed_dim)
     self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
     self.positional_encoding = positionalEncoding(embed_dim, max_seq_length)
     self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, heads, dff, dropout) for i in range(num_layers)])
     self.decoder_layers = nn.ModuleList([DecoderLayer(embed_dim, heads, dff, dropout) for i in range(num_layers)])

     self.fc = nn.Linear(embed_dim, tgt_vocab_size)
     self.dropout = nn.Dropout(dropout)

  def generate_mask(self, src, tgt):
    # src_mask: [batch, 1, 1, src_seq_len]
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    # tgt_mask: [batch, 1, tgt_seq_len, tgt_seq_len]
    seq_length = tgt.size(1)
    nopeak_mask = torch.tril(torch.ones((seq_length, seq_length), device=tgt.device)).bool()

    # Combined padding and look-ahead mask
    tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = tgt_pad_mask & nopeak_mask

    return src_mask, tgt_mask

  def forward(self, src, tgt):
    src_mask, tgt_mask = self.generate_mask(src, tgt)

    # Encoder pipeline
    src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
    enc_output = src_embedded
    for enc_layer in self.encoder_layers:
        enc_output = enc_layer(enc_output, src_mask)

    # Decoder pipeline
    dec_output = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
    for dec_layer in self.decoder_layers:
        # FIXED: Pass dec_output as the first argument
        dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

    output = self.fc(dec_output)
    return output
