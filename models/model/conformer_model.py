#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 12:04:03 2023

@author: ahannan
"""

import torch
import torchaudio
from torch import nn
from torch import Tensor
from models.model.conformer import Conformer
from models.embedding.positional_encoding import PositionalEncoding

torch.set_printoptions(profile='full')        
class Conv1dSubampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: list) -> None:
        super(Conv1dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels[0], kernel_size=3, stride=2, padding=1, padding_mode='zeros'),
            nn.GELU(),
            nn.BatchNorm1d(out_channels[0]),
            nn.Conv1d(out_channels[0], out_channels[1], kernel_size=3, stride=2, padding=0, padding_mode='zeros'),       # padding = 1
            nn.GELU(),
            # nn.BatchNorm1d(out_channels[1]),
            # nn.Conv1d(out_channels[1], out_channels[2], kernel_size=3, stride=2, padding=1, padding_mode='zeros'),
            # nn.GELU(),
            # nn.BatchNorm1d(out_channels[2]),
        )
    def forward(self, inputs: Tensor) -> torch.tensor:
        outputs = self.sequential(inputs)
        return outputs

class Conformer_model(nn.Module):
    ### This model contains single linear layer as a decoder at the end of last encoder
    def __init__(self, conv_filters,
                       src_pad_idx, 
                       n_enc_replay, 
                       enc_voc_size, 
                       dec_voc_size, 
                       d_model, 
                       n_head, 
                       max_len, 
                       dim_feed_forward, 
                       n_encoder_layers, 
                       features_length, 
                       drop_prob, 
                       depthwise_kernel_size, 
                       device, 
                       flag_specAug : bool = False):
        """ Flag_use_single_out: If False, uses linear decoders as specified at multiple Early-Exits, 
                                if True, uses Only ONE Linear Decoder at the end of model
            Flag_use_gating: if False, no Gating is applied to the model,
                            If True, Gating Mechanism is applied to the model. """
        super().__init__()
        self.input_dim = d_model
        self.num_heads = n_head
        self.ffn_dim = dim_feed_forward
        self.num_layers = n_encoder_layers
        self.depthwise_conv_kernel_size = depthwise_kernel_size
        self.n_enc_replay = n_enc_replay
        self.dropout = drop_prob
        self.device = device
        self.specAug = flag_specAug
        self.time_mask_value = 64                 # Maximum 25% of the Time-steps would be masked
        self.freq_mask_value = 27                 # Maximum 27 mel-feature banks filters would be masked
        
        self.conv_subsample = Conv1dSubampling(in_channels = features_length, out_channels = conv_filters)
        self.positional_encoder = PositionalEncoding(d_model=d_model, dropout=drop_prob, max_len=max_len)
        self.single_linear = nn.Linear(d_model, dec_voc_size)
        self.conformer = nn.ModuleList([Conformer(input_dim=self.input_dim, 
                                        num_heads=self.num_heads, 
                                        ffn_dim=self.ffn_dim, 
                                        num_layers=self.num_layers, 
                                        depthwise_conv_kernel_size=self.depthwise_conv_kernel_size, 
                                        dropout=self.dropout) for _ in range(self.n_enc_replay)])

    def get_gate_values(self, flag_print = False):
        g_val = []
        for i in range(len(self.conformer)):
            for j in range(2):
                g_val.append(self.conformer[i].conformer_layers[j].gate.data.item())
        
        if flag_print:
            print("Current gate values are ", g_val)
        
        return g_val

    def forward(self, src, lengths):
        # Input Size = [1, 80, x - number_of_spectrograms/or something else]
        
        # Whether to perform Spec Augmentation
        if self.specAug:
            t_mask = torchaudio.transforms.TimeMasking(time_mask_param = self.time_mask_value, iid_masks = False, p = 1.0)
            f_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param = self.freq_mask_value, iid_masks = False)
            
            src = t_mask(src)
            src = f_mask(src) 
        
        #convolution        
        out = self.conv_subsample(src)           # Size = [1, 256 + 12 (number of gates), y] where y = x - 4 / 4
        # print("Feature Extractor's size: ", out.size())

        out = self.positional_encoder(out.permute(0,2,1))

        length = torch.clamp(lengths/4, max = out.size(1)).to(torch.int).to(self.device)
        # print("Length is: ", length.size())
        
        enc_out = []
        for i, layer in enumerate(self.conformer):      # len(self.conformer) = 6 ---- (0, 1, 2, 3, 4, 5)
            # enc, _ = layer(enc, length)
            out, _ = layer(out, length)
            enc_out.append(out)
            
        out = self.single_linear(out)
        # Size = [1, y, 256]
        # print("After Linear shape is : ", out.size())
        out = torch.nn.functional.log_softmax(out, dim = 2)
        # Size = [1, y, 256]
        # print("After Log-Softmax shape is : ", out.size())
        
        return enc_out[-1], out
        
