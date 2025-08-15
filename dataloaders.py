#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:59:24 2023

@author: ahannan
"""

import torch
import torchaudio

from conf import *
from util.data_loader import collate_fn, collate_infer_fn

train_dataset1 = torchaudio.datasets.LIBRISPEECH(dataset_path, 
                                                 url = "train-clean-100", 
                                                 download = False)
train_dataset2 = torchaudio.datasets.LIBRISPEECH(dataset_path, 
                                                 url="train-clean-360", 
                                                 download = False)
train_dataset3 = torchaudio.datasets.LIBRISPEECH(dataset_path, 
                                                 url="train-other-500", 
                                                 download = False)

train_dataset = torch.utils.data.ConcatDataset([train_dataset1,train_dataset2,train_dataset3])       # to use Libri-1000

train_loader = torch.utils.data.DataLoader(train_dataset, pin_memory = False, 
                                          batch_size = batch_size, shuffle = shuffle, 
                                          collate_fn = collate_fn, num_workers = train_num_workers)

#############################################################################################################################

valid_dataset = torchaudio.datasets.LIBRISPEECH(dataset_path,
                                                 url = "dev-clean", 
                                                 download = False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, pin_memory=False, 
                                          batch_size = batch_size, shuffle = shuffle, 
                                          collate_fn = collate_fn, num_workers = test_num_workers)

#############################################################################################################################

test_dataset = torchaudio.datasets.LIBRISPEECH(dataset_path,
                                                 url = "test-clean", 
                                                 download = False)
test_loader = torch.utils.data.DataLoader(test_dataset, pin_memory=False, 
                                          batch_size = 1, shuffle = False, 
                                          collate_fn = collate_infer_fn, num_workers = test_num_workers)


