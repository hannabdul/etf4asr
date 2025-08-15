"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import sys
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
#from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device, gate_prob = 1.0):
        super().__init__()
        #self.emb = TransformerEmbedding(d_model=d_model,
        #                                max_len=max_len,
        #                                vocab_size=enc_voc_size,
        #                                drop_prob=drop_prob,
        #                                device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model = d_model,
                                                  ffn_hidden = ffn_hidden,
                                                  n_head = n_head,
                                                  drop_prob = drop_prob,
                                                  device = device,
                                                  gate_prob = gate_prob)
                                     for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, s_mask):

        for layer in self.layers:
            x = layer(x, s_mask)

        x = self.layer_norm(x)
        
        return x
