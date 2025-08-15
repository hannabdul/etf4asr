"""##############################################################################################################################################
------------------------------- Please Read before working with the script -----------------------------------
Current setting is for 6 Conformers, meaning You have done Encoder Representation Learning (Pre-Training) for a model with 6 conformer modules first, and then fine-tuned it. To evaluate the final model after fine-tuning, this configuration will work. 
If You have have performed Encoder Representation Learning for for 6 Conformer modules, and want to finetune for 4 or 2 conformer modules, subtract the value from the n_enc_replay argument of the model below. Make sure the instance of model class has similar arguments in (Encoder Representation Learning Phase -- pretrain_for_ASR.py), (Finetuning phase -- train.py) 
#################################################################################################################################################"""

# Pytorch
import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from conf import *
from transforms import *
from dataloaders import *

from models.model.conformer_model import Conformer_model
from models.model_functions import count_parameters
from NoamOpt import NoamOpt
from util.data_loader import text_transform
from util.beam_infer import ctc_predict
from util.data_loader import collate_fn

from torch.utils.tensorboard import SummaryWriter

# Other
import json
import argparse
import os
import numpy as np

# Sentencepiece
import sentencepiece as spm

def kld_loss(pred, target):
    kl_loss = nn.KLDivLoss(reduction = "batchmean", log_target = True)
    error = kl_loss(pred, target)
    return error

def mae_loss(pred, target):
    error = torch.abs(target - pred).mean()
    # print(error)
    return error

def mse_loss(pred, target):
    error = (target - pred)**2
    error = error.mean()
    return error

def clip_loss(pred, target):
    pred = pred.mean(dim=1)
    target = target.mean(dim=1)
    logits = (pred @ target.T) * torch.exp(torch.tensor(np.log(1./0.07))).to(pred.device)
    labels = torch.arange(pred.shape[0], device=pred.device)
    loss_p = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    loss = (loss_t + loss_p) / 2.
    return loss

####################################################################
if __name__ == "__main__":     
	  # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    writer = SummaryWriter(log_path)
    torch.manual_seed(1234)
    torch.set_printoptions(precision = 4)
    
    # Model_orig is the model with 12 (n_enc_replay * 2) attention_blocks
    model_orig = Conformer_model(conv_filters = conv_filters, 
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
                        
    # Model_lr is the model with 6 (n_enc_replay) attention_blocks -- Default
    model_lr = Conformer_model(conv_filters = conv_filters, 
                        src_pad_idx = src_pad_idx,
                        n_enc_replay = n_enc_replay,                               # Original Value of n_enc_replay = 6
                        d_model = d_model,
                        enc_voc_size = enc_voc_size,
                        dec_voc_size = dec_voc_size,
                        max_len = max_len,
                        dim_feed_forward = dim_feed_forward,
                        n_head = n_heads,
                        n_encoder_layers = n_encoder_layers // 2,                  # Original Value of n_encoder_layers = 2
                        features_length = n_mels,
                        drop_prob = drop_prob,
                        depthwise_kernel_size = depthwise_kernel_size,
                        device = device,
                        flag_specAug = False).to(device)
    print(model_lr)
    print(f'The model has {count_parameters(model_lr):,} trainable parameters')
    print("batch_size : ",batch_size, " num_heads : ",n_heads, " num_encoder_layers : ", n_encoder_layers, " optimizer : ", "NOAM[warmup ", warmup, "]", "vocab_size : ", dec_voc_size, 
      "SOS, EOS, PAD ", trg_sos_idx, trg_eos_idx, trg_pad_idx, "data_loader_len : ", len(train_loader), "DEVICE : ", device, "Using SpecAug", model_lr.specAug) 
    message = model_orig.load_state_dict(torch.load(os.path.join(base_path, "trained_model", "baseline_libri1000", "Conf_mod150-tc1000.pt"), map_location = device))
    print("Loading Full / Original Model Weights ---- ", message)

    for name, param in model_orig.named_parameters():
		    param.requires_grad = False

    for name, param in model_lr.named_parameters():
		    param.requires_grad = True
    
    optimizer = AdamW(params = model_lr.parameters(), lr = 0.0, betas = (adam_b1, adam_b2), eps = adam_eps, weight_decay = weight_decay)
    scheduler = NoamOpt(d_model, warmup, optimizer)

		# Mixed Precision Gradient Scaler
    scaler = torch.cuda.amp.GradScaler(enabled = grad_scaler)     # training_params["mixed_precision"] = True in Efficient Conformer Paper

    optimizer.zero_grad()
    
    print("################ Training for {} Epochs --- Saving at {} ".format(total_epochs, project_name))
    
    for epoch in range(total_epochs):
        model_lr.train()
        epoch_loss = 0.0
        start_time = time.time()
        # if os.path.join(model_save_path, "Pretrain_Conf_mod{:03d}-tc1000".format(epoch)):
        #     message = model_lr.load_state_dict(torch.load(os.path.join(model_save_path, "Pretrain_Conf_mod{:03d}-tc1000".format(epoch)), map_location = device))
        #     print(message)

        # # Epoch training
        for step, batch in enumerate(train_loader):
            if not batch:
                continue

            src = batch[0].to(device) 
            trg = batch[1][:,:-1].to(device)             # cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28] 
            trg_expect = batch[1][:,1:].to(device)       # shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]   
            # print("INPUT:",text_transform.int_to_text(trg[0]))
            valid_lengths = batch[3]
                        
            # Automatic Mixed Precision Casting (model prediction + loss computing)
            with torch.cuda.amp.autocast(enabled = grad_scaler):           # training_params["mixed_precision"] = True in Efficient Conformer Paper
            
                pred_orig = model_orig(src, valid_lengths)
                pred_lr = model_lr(src, valid_lengths)
                				    
                loss = 0
                c_loss = 0
                m_loss = 0
                
                ############################################################################################
                #### Select the desired loss function
                c_loss = clip_loss(pred_lr[-2], pred_orig[-2])
                # m_loss = mae_loss(pred_lr[-1], pred_orig[-1])
                m_loss = mse_loss(pred_lr[-1], pred_orig[-1])
                # k_loss = kld_loss(pred_lr[-1], pred_orig[-1])
                loss = c_loss + m_loss
                # loss = c_loss
                # loss = m_loss
                # loss = k_loss
                ############################################################################################
                
                writer.add_scalar('Clip Loss', c_loss.item())
                # writer.add_scalar('MAE Loss', m_loss.item())
                writer.add_scalar('MSE Loss', m_loss.item())
                # writer.add_scalar('KLD Loss', k_loss.item())
                writer.add_scalar('Total Loss', loss.item())
				        			
			      # Accumulate gradients
            scaler.scale(loss).backward()
            nn.utils.clip_grad_value_(model_lr.parameters(), clip_value = 1.0)
            
            # Update Epoch Variables
            
            epoch_loss += loss.item()

			      # if accum_iter < args.accum_iter:
			      #     continue
            
            if step % 100 == 0:
                print('Loss {0}/{1}: {2}'.format(step, epoch+1, loss))

			      # Update Parameters, Zero Gradients and Update Learning Rate
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()      

        print('######### Epoch Loss {0}: {1}'.format(epoch+1, epoch_loss/step))
        writer.add_scalar('Epoch Loss', epoch_loss/step)
        
        ############### Saving State_dict of each "model" and "optimizer" 
        best_model = os.path.join(model_save_path, "Pretrain_Conf_mod{:03d}-tc1000".format(epoch + 1))
        print("saving:",best_model)
        torch.save(model_lr.state_dict(), best_model)
        lrate = os.path.join(model_save_path, "Pretrain_Conf_lr{:03d}-tc1000".format(epoch + 1))
        print("Saving:", lrate)
        torch.save(optimizer.state_dict(), lrate)
        print("Time per epoch : ", time.time() - start_time, " seconds")
        
        if epoch_loss <= 1e-2:
            writer.flush()
            print("closing the Writer...!!")
            break
            
    writer.flush()
    print("closing the Writer...!!")          
