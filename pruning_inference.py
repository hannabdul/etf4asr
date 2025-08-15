import os
import time
import torch
import torchaudio

from conf import *
from pruning import model_pruning
from models.model_functions import count_parameters
from models.model.conformer_model import Conformer_model

from transforms import *
from util.beam_infer import ctc_predict
from util.data_loader import collate_infer_fn
from util.tokenizer import apply_lex, load_dict

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
                        n_encoder_layers = n_encoder_layers,
                        features_length = n_mels,
                        drop_prob = drop_prob,
                        depthwise_kernel_size = depthwise_kernel_size,
                        device = device,
                        flag_specAug = False).to(device)

pruning_amount = 0.225

def get_desired_keys(in_model):
    weights_list = []
    biases_list = []
    for key in in_model.state_dict():
        if "ffn1" in key or "ffn2" in key or "self_attn.in_proj" in key or "self_attn.out_proj" in key:
        # if "ffn1" in key or "ffn2" in key or "conv_module" in key or "self_attn.in_proj" in key or "self_attn.out_proj" in key:
            if "weight" in key:
                weights_list.append(key)
            if "bias" in key:
                biases_list.append(key)
    return weights_list, biases_list

def evaluate(model):
    
    file_dict = os.path.join(base_path, "librispeech.lex")
    words = load_dict(file_dict)
     
    model.eval()
    
    set_ = "test-clean"
    if set_ == "test-clean":
    # for set_ in "test-clean","test-other","dev-clean", "dev-other":
        print(set_, ", Using Device: ", device)
        
        test_dataset = torchaudio.datasets.LIBRISPEECH("/stek/falavi/corpora/", 
                                                       url = set_, 
                                                       download = False)
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  pin_memory = False, 
                                                  batch_size = 1, 
                                                  num_workers = num_workers, 
                                                  shuffle = False, 
                                                  collate_fn = collate_infer_fn)
        total_time = 0

        for it, batch in enumerate(test_loader):
            t_start = time.time()
                        
            trg_expect = batch[1][:,1:].to(device) #shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]   
            trg = batch[1][:,:-1].to(device) #cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28] 
            for trg_expect_ in trg_expect:
                if bpe_flag == True:
                    print(set_,"EXPECTED:",sp.decode(trg_expect_.squeeze(0).tolist()).lower())
                else:                    
                    print(set_,"EXPECTED:",re.sub(r"[#^$]+","",text_transform.int_to_text(trg_expect_.squeeze(0))))
            valid_lengths = batch[2]
            
            model_inp = batch[0].to(device)
            _, encoder = model(model_inp, valid_lengths)
            # encoder = (1 / 0.6) * encoder    # For Baseline experiment (12 conformers)
            # encoder = (1 / 0.334) * encoder    # For 6 conformers
            encoder = (1 / 0.167) * encoder   # For 4 Conformers
                        
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
###############################################################################
if __name__ == '__main__':
    ## Model Trained on Libri-1000 with Batch_Gating of 0.8
    test_model_path = os.path.join(base_path, "trained_model", "baseline_libri1000_specAug", "Conf_mod150-tc1000.pt")
    
    # test_model_path = os.path.join(base_path, "trained_model", "ft_libri1000_confs_6", "Conf_mod050-tc1000.pt")
    # test_model_path = os.path.join(base_path, "trained_model", "ft_libri1000_confs_4", "Conf_mod050-tc1000.pt")
    # test_model_path = os.path.join(base_path, "trained_model", "ft_libri1000_confs_2", "Conf_mod050-tc1000.pt")
    # test_model_path = os.path.join(base_path, "trained_model", "pretrain_4_ft_libri1000_confs_4", "Conf_mod050-tc1000.pt")
    
    # test_model_path = os.path.join(base_path, "trained_model", "ft_libri1000_confs_6_clip_only", "Conf_mod050-tc1000.pt")      # Pre-trained using Clip Only Loss
    # test_model_path = os.path.join(base_path, "trained_model", "ft_libri1000_confs_6_mae_only", "Conf_mod050-tc1000.pt")      # Pre-trained using MAE Only Loss
    # test_model_path = os.path.join(base_path, "trained_model", "ft_libri1000_confs_6_mse_only", "Conf_mod050-tc1000.pt")      # Pre-trained using MSE Only Loss
    # test_model_path = os.path.join(base_path, "trained_model", "ft_libri1000_confs_6_clip_mae_loss", "Conf_mod050-tc1000.pt")        # Pre-trained using Clip + MAE Loss

    # del model.conformer[0]
    # del model.conformer[0]
    # del model.conformer[0]
    # del model.conformer[0]
    
    model.load_state_dict(torch.load(test_model_path, map_location = device, weights_only = True))
    
    desired_weights, desired_biases = get_desired_keys(model)
    pruned_params = []
    for i, (w_key, b_key) in enumerate(zip(desired_weights, desired_biases)):
        temp, _ = model_pruning(model = model, module_key = w_key, amount = pruning_amount, substitute_module = True, flag_print = True)
        temp1, _ = model_pruning(model = model, module_key = b_key, amount = pruning_amount, substitute_module = True, flag_print = True)
        pruned_params.append(temp)
        pruned_params.append(temp1)
        del temp, temp1
        
    print("The percentage of Neurons Pruned is : ", (sum(pruned_params) / count_parameters(model)) * 100, "%")

    evaluate(model)
    
    
