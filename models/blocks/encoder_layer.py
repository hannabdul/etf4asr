"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
import sys
import torch
import torch.nn as nn

#from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, device, gate_prob = 1.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model) #LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model) #LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        
        self.gate = torch.bernoulli(torch.randint(1, (1, 1)), p = gate_prob).item()#.to(device)

    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x
        x = self.norm1(x) #norm first
        x = self.attention(q=x, k=x, v=x, mask=s_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = (self.gate * x) + _x #self.norm1(x + _x)
        # x = x + _x
        
        # 3. positionwise feed forward network
        _x = x
        x = self.norm2(x)
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = (self.gate * x) + _x #self.norm2(x + _x)
        # x = x + _x
        return x
