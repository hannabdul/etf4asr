#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:53:34 2023

@author: ahannan
"""

import torch
from torch.nn.utils import prune

def model_pruning(model, module_key, amount, substitute_module = False, flag_print = False):
    """ Sub_module parameter sustitutes the original tensor with Pruned Tensor """
    module = model.state_dict()[module_key]
    params_bef_pru = module.size().numel()

    pruning = prune.L1Unstructured(amount)

    module = pruning.prune(module)
    params_aft_pru = torch.count_nonzero(module).item()
    if flag_print:
        print('Neurons pruned at layer ', module_key, ': ', params_bef_pru - params_aft_pru, '  ---  Total Neurons before Pruning: ', params_bef_pru)

    if substitute_module:
        model.state_dict()[module_key].copy_(module)    # substituing the pruned weights

    return params_bef_pru - params_aft_pru, module