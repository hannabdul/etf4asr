#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:44:11 2023

@author: ahannan
"""

import re
import os
import sentencepiece as spm

bpe_flag = True
flag_use_single_out = True

dataset_path = "/stek/corpora/"
dataset_to_use = "LibriSpeech"                   #### Or Tedlium-v3
project_name = "sample_pretrain_" + dataset_to_use
# project_name = "sample_ft_" + dataset_to_use

base_path = os.getcwd()
log_path = os.path.join(base_path, "runs", project_name)
os.makedirs(log_path, exist_ok = True)
model_save_path = os.path.join(base_path, "trained_model", project_name)
os.makedirs(model_save_path, exist_ok = True)

############ DataLoader Settings
train_num_workers = 12
test_num_workers = 8
shuffle = True

sample_rate = 16000
n_fft = 512
win_length = 320            # 20 ms Segments
hop_length = 160            # 10 ms Overlap
n_mels = 80
n_mfcc = 80

############ model parameter setting
batch_size = 64
max_len = 2000
conv_filters = [256, 256]
d_model = 256
n_encoder_layers = 2
n_decoder_layers = 6
n_heads = 4
n_enc_replay = 6
expansion_factor = 4
dim_feed_forward= d_model * expansion_factor
drop_prob = 0.1
depthwise_kernel_size = 31
max_utterance_length = 401

########### Token numbers when BPE_Flag is False.
src_pad_idx = 0
trg_pad_idx = 30
trg_sos_idx = 1
trg_eos_idx = 31
enc_voc_size = 32     
dec_voc_size = 32      

sp = spm.SentencePieceProcessor()
if bpe_flag == True:
    sp.load(os.path.join(base_path, "libri.bpe-256.model"))
    src_pad_idx = 0
    trg_pad_idx = 126
    trg_sos_idx = 1
    trg_eos_idx = 2
    enc_voc_size = sp.get_piece_size()
    dec_voc_size = sp.get_piece_size()
    lexicon = os.path.join(base_path, "librispeech-bpe-256.lex")
    tokens = os.path.join(base_path, "librispeech-bpe-256.tok")


# optimizer parameter setting
factor = 0.9
adam_b1 = 0.9
adam_b2 = 0.98
adam_eps = 1e-9
warmup = 10000
grad_scaler = False

total_epochs = 50          # 100 - for pretraining      # 50 - for fine-tuning       # 150 - baseline
clip = 1.0
weight_decay = 1e-6
inf = float('inf')
