#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:26:43 2023

@author: ahannan
"""

import os
import io
import time
import torchaudio
# import editdistance
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder

from conf import *
from transforms import *
from util.data_loader import collate_infer_fn
from util.tokenizer import apply_lex, load_dict
from util.beam_infer import ctc_predict, beam_predict, avg_models

# This is for visualizing the beam scores
from util.beam_infer import get_ctc_scores

encoder_output = []           # appending the output of last encoder layer

torch.set_printoptions(precision=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

def evaluate(model, gate_prob = 1.0):

    file_dict = 'librispeech.lex'
    words=load_dict(file_dict)

    # path = os.getcwd()+'/trained_model/'
    # model = avg_models(model, path, 238, 248)        
    model.eval()
    
    gum_gates = None
    #w_ctc = float(sys.argv[1])


    # beam_size = 10                  # This Variable is not used
    batch_size = 1
    
    set_ = "test-clean"
    if set_ == "test-clean":
    # for set_ in "test-clean","test-other","dev-clean", "dev-other":
        print(set_, ", Using Device: ", device)        
        test_dataset = torchaudio.datasets.LIBRISPEECH("/stek/falavi/corpora/", url=set_, download=False)
        data_loader = torch.utils.data.DataLoader(test_dataset, pin_memory=False, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=collate_infer_fn)
        total_time = 0
        # stacked_emissions = []
        
        # for batch in data_loader: 
        for it, batch in enumerate(data_loader):
            t_start = time.time()
            
            trg_expect =batch[1][:,1:].to(device) #shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]   
            trg = batch[1][:,:-1].to(device) #cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28] 
            for trg_expect_ in trg_expect:
                if bpe_flag == True:
                    print(set_,"EXPECTED:",sp.decode(trg_expect_.squeeze(0).tolist()).lower())
                else:                    
                    print(set_,"EXPECTED:",re.sub(r"[#^$]+","",text_transform.int_to_text(trg_expect_.squeeze(0))))
            valid_lengths=batch[2]

            encoder, curr_gates = model(batch[0].to(device), valid_lengths)
            
            ## Scale Here
            # encoder = (1 / gate_prob) * encoder
            encoder = 1 / 0.5 * (encoder)
            ##
            # stacked_emissions.append(encoder[0].squeeze().detach().cpu().numpy())
            # stacked_emissions.append(get_ctc_scores(encoder[0], 0))
            

            if it == 0:
                gum_gates = curr_gates
            else:
                gum_gates = torch.cat((gum_gates, curr_gates), 1)
            del curr_gates
            if not flag_use_single_out:    # Use this if the model is trained with early-exits and you desire the output of only last exit
                i = 0
                for enc in encoder:
                    i = i + 1
                    best_combined = ctc_predict(enc, i - 1)
                    for best_ in best_combined:
                        if bpe_flag==True:
                            print(set_," BEAM_OUT_",i,":",  apply_lex(sp.decode(best_).lower(),words))
                        else:
                            print(set_," BEAM_OUT_",i,":",  apply_lex(re.sub(r"[#^$]+","",best_.lower()),words))
                t_end = time.time()
                # print("Total Time taken per batch: ", t_end - t_start)
                total_time = total_time + (t_end - t_start)
                # print("Total Time Taken: ", total_time)
            else:
                i = 1
                best_combined = ctc_predict(encoder[0], i - 1)
                # best_scores = get_ctc_scores(encoder[0], i - 1)
                # stacked_emissions.append(best_scores)
                # print(best_scores)
                for best_ in best_combined:
                    if bpe_flag == True:
                        print(set_," BEAM_OUT_",i,":",  apply_lex(sp.decode(best_).lower(),words))
                    else:
                        print(set_," BEAM_OUT_",i,":",  apply_lex(re.sub(r"[#^$]+","",best_.lower()),words))
                t_end = time.time()
                # print("Total Time taken per batch: ", t_end - t_start)
                total_time = total_time + (t_end - t_start)
                # print("Total Time Taken: ", total_time)
                
            # if flag_use_batch_gating:
            #     model.update_gate_prob(new_prob = gate_prob, flag_print = False)
            
            # if it >= 0.001 * len(data_loader):     # For 8% Dataset
            # if it == 2:
            #     print("Exiting Now..!!")
            #     break
            
        print("Total Time Taken: ", total_time, ' seconds')
        # return stacked_emissions        
    return gum_gates


            # # Computes the output after each exit in the early-exit model
            # for enc in encoder:
            #     i=i+1
            #     best_combined = ctc_predict(enc, i - 1)
            #     for best_ in best_combined:
            #         if bpe_flag==True:
            #             print(set_," BEAM_OUT_",i,":",  apply_lex(sp.decode(best_).lower(),words))
            #         else:
            #             print(set_," BEAM_OUT_",i,":",  apply_lex(re.sub(r"[#^$]+","",best_.lower()),words))    

