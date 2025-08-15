"""##############################################################################################################################################
------------------------------- Please Read before working with the script -----------------------------------
Current setting is for 6 Conformers, meaning You have done Encoder Representation Learning (Pre-Training) for a model with 6 conformer modules first, and then fine-tuned it. To evaluate the final model after fine-tuning, this configuration will work. 
If You have have performed Encoder Representation Learning for for 6 Conformer modules, and want to finetune for 4 or 2 conformer modules, subtract the value from the n_enc_replay argument of the model below. Make sure the instance of model class has similar arguments in (Encoder Representation Learning Phase -- pretrain_for_ASR.py), (Finetuning phase -- train.py) 
#################################################################################################################################################"""

import os
import time
import torch
import torchaudio

from conf import *
from models.model.conformer_model import Conformer_model

from transforms import *
from util.beam_infer import ctc_predict
from util.data_loader import collate_infer_fn
from util.tokenizer import apply_lex, load_dict
from models.model_functions import count_parameters

torch.set_printoptions(precision=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1234)

model = Conformer_model(conv_filters = conv_filters, 
                        src_pad_idx = src_pad_idx,
                        n_enc_replay = n_enc_replay,
                        d_model = d_model,
                        enc_voc_size = enc_voc_size,
                        dec_voc_size = dec_voc_size,
                        max_len = max_len,
                        dim_feed_forward = dim_feed_forward,
                        n_head = n_heads,
                        n_encoder_layers = n_encoder_layers // 2,
                        features_length = n_mels,
                        drop_prob = drop_prob,
                        depthwise_kernel_size = depthwise_kernel_size,
                        device = device,
                        flag_specAug = False).to(device)

def evaluate(model):
    
    file_dict = os.path.join(base_path, "librispeech.lex")
    words = load_dict(file_dict)
     
    model.eval()
    total_time = 0

    for it, batch in enumerate(test_loader):
        t_start = time.time()
                        
        trg_expect = batch[1][:,1:].to(device)        # shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]   
        trg = batch[1][:,:-1].to(device)              # cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28] 
        for trg_expect_ in trg_expect:
            if bpe_flag == True:
                print(set_,"EXPECTED:",sp.decode(trg_expect_.squeeze(0).tolist()).lower())
            else:                    
                print(set_,"EXPECTED:",re.sub(r"[#^$]+","",text_transform.int_to_text(trg_expect_.squeeze(0))))
        valid_lengths = batch[2]
            
        model_inp = batch[0].to(device)
        _, encoder = model(model_inp, valid_lengths)
        ############ Either scale the logits here,, or go to utils.beam_infer.py file and change the "word_score = -6.0" on line-23
        encoder = (1 / 0.6) * encoder
        
        best_combined = ctc_predict(encoder)

        for best_ in best_combined:
            if bpe_flag == True:
                print(set_," BEAM_OUT_ 1", ":",  apply_lex(sp.decode(best_).lower(),words))
            else:
                print(set_," BEAM_OUT_ 1", ":",  apply_lex(re.sub(r"[#^$]+","",best_.lower()),words))
        t_end = time.time()
        # print("Total Time taken per batch: ", t_end - t_start)
        total_time = total_time + (t_end - t_start)
          
    print("Total Time Taken: ", total_time, ' seconds')

###############################################################################
# print(model)
# print("Total Params: ", count_parameters(model))
###############################################################################

if __name__ == '__main__':
    ## Load the trained model that you want to perform inference on
    test_model_path = os.path.join(base_path, "trained_model", "sample_pretrain_LibriSpeech", "Conf_mod050-tc1000.pt")
    model.load_state_dict(torch.load(test_model_path, map_location = device, weights_only = True))
    evaluate(model)
    
    
