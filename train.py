"""##############################################################################################################################################
------------------------------- Please Read before working with the script -----------------------------------
Current setting is for 6 Conformers, meaning You have done Encoder Representation Learning (Pre-Training) for a model with 6 conformer modules first, and then fine-tuned it. To evaluate the final model after fine-tuning, this configuration will work. 
If You have have performed Encoder Representation Learning for for 6 Conformer modules, and want to finetune for 4 or 2 conformer modules, subtract the value from the n_enc_replay argument of the model below. Make sure the instance of model class has similar arguments in (Encoder Representation Learning Phase -- pretrain_for_ASR.py), (Finetuning phase -- train.py) 
#################################################################################################################################################"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
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

writer = SummaryWriter(log_path)

torch.manual_seed(1234)
torch.set_printoptions(precision = 4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Conformer_model(conv_filters = conv_filters, 
                        src_pad_idx=src_pad_idx,
                        n_enc_replay=n_enc_replay,
                        d_model=d_model,
                        enc_voc_size=enc_voc_size,
                        dec_voc_size=dec_voc_size,
                        max_len=max_len,
                        dim_feed_forward=dim_feed_forward,
                        n_head=n_heads,
                        n_encoder_layers = n_encoder_layers // 2,
                        features_length=n_mels,
                        drop_prob=drop_prob,
                        depthwise_kernel_size=depthwise_kernel_size,
                        device=device,
                        flag_specAug = False).to(device)
                        
print(model)

print(f'The model has {count_parameters(model):,} trainable parameters')
print("batch_size : ",batch_size, " num_heads : ",n_heads, " num_encoder_layers : ", n_encoder_layers, " optimizer : ", "NOAM[warmup ", warmup, "]", "vocab_size : ", dec_voc_size, 
      "SOS, EOS, PAD ", trg_sos_idx, trg_eos_idx, trg_pad_idx, "data_loader_len : ", len(train_loader), "DEVICE : ", device, "Using SpecAug", model.specAug)

ctc_loss = nn.CTCLoss(blank = 0, zero_infinity = True)
optimizer = NoamOpt(d_model, warmup, AdamW(params = model.parameters(), lr = 0.0, betas = (adam_b1, adam_b2), eps = adam_eps, weight_decay = weight_decay))

def validate(model, iterator):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].to(device) 
            trg = batch[1][:,:-1].to(device)
            trg_expect = batch[1][:,1:].to(device) #shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]
            
            valid_lengths = batch[3]

            _, encoder = model(src, valid_lengths)
            
            ctc_target_len = batch[2]
            
            ctc_input_len = torch.full(size=(encoder.size(0),), fill_value = encoder.size(1), dtype=torch.long)
            
            loss = ctc_loss(encoder.permute(1, 0, 2), batch[1], ctc_input_len, ctc_target_len).to(device)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def train(model, iterator, ep_num):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        if not batch:
            continue

        src = batch[0].to(device) 
        trg = batch[1][:,:-1].to(device)                  # cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28] 
        trg_expect = batch[1][:,1:].to(device)            # shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]   
        #print("INPUT:",text_transform.int_to_text(trg[0]))
        valid_lengths = batch[3]

        _, encoder = model(src, valid_lengths)
        # print("Encoder Size is: ", encoder.size())

        ctc_target_len = batch[2]
        # print(ctc_target_len, '---', ctc_target_len.size())

        if i % 300 == 0:
            if bpe_flag==True:
                print("EXPECTED:", sp.decode(trg_expect[0].tolist()).lower())
            else:
                print("EXPECTED:", text_transform.int_to_text(trg_expect[0]))
        
           
        ctc_input_len = torch.full(size=(encoder.size(0),), fill_value = encoder.size(1), dtype=torch.long)
        # print(encoder.size(), ctc_input_len)

        loss = ctc_loss(encoder.permute(1, 0, 2), batch[1], ctc_input_len, ctc_target_len).to(device)
                    
        if i % 300 == 0:
            if bpe_flag == True:
                print("CTC_OUT at [", i, "]:", sp.decode(ctc_predict(encoder[0].unsqueeze(0))).lower())
            else:
                print("CTC_OUT at [", i, "]:", ctc_predict(encoder))
                        
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        writer.add_scalar('CTC loss', loss.item())
        print('Epoch: ', ep_num, ', step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item()) #, "LR: ", optimizer.param_groups[0]['lr'])
        
        epoch_loss += loss.item()    
    return epoch_loss / len(iterator)   

def run(model, total_epoch, best_loss):
    prev_loss = 9999999
    nepoch = 0
    
    # # Fine-tuning the model of 6 Conformers for 50 epochs on libri-1000
    message = model.load_state_dict(torch.load("/stek/ahannan/interspeech25/new_model/trained_model/pretrain_libri1000_confs_6/Pretrain_Conf_mod100-tc1000", 
                                   map_location = device, weights_only = True))
    
    
    print("################ Training for {} Epochs --- Saving at {} ".format(total_epochs, project_name))
    
    best_model = os.path.join(model_save_path, "Conf_mod{:03d}-tc1000".format(nepoch))
    best_lr = os.path.join(model_save_path, "Conf_lr{:03d}-tc1000".format(nepoch))

    if os.path.exists(best_model):
        print('loading model checkpoint: ', best_model)
        model.load_state_dict(torch.load(best_model, map_location=device))

    if os.path.exists(best_lr):
        print('loading learning rate checkpoint: ', best_lr)
        optimizer.load_state_dict(torch.load(best_lr))         
  
    for step in range(nepoch, total_epoch):
        start_time = time.time()
          
        train_loss = train(model = model,               # Input Model
                           iterator = train_loader,     # Dataloader object for LibriSpeech dataset
                           ep_num = step                # Current Epoch
                           )
        valid_loss = validate(model = model,
                              iterator = valid_loader)                      
        print("Train_Loss at epoch {} = {}".format(step, train_loss))
        print("Validation_Loss at epoch {} = {}".format(step, valid_loss))
        print("--------------------------------------------------")
        
        writer.add_scalar("Training Loss/Epoch", train_loss, step)
        writer.add_scalar("Valid Loss/Epoch", valid_loss, step)
        
        ################# Saving each checkpoint
        if train_loss:
            prev_loss = train_loss
            best_model = os.path.join(model_save_path, "Conf_mod{:03d}-tc1000.pt".format(step + 1))
            print("saving:",best_model)
            torch.save(model.state_dict(), best_model)
            lrate = os.path.join(model_save_path, "Conf_lr{:03d}-tc1000.pt".format(step + 1))
            print("Saving:", lrate)
            torch.save(optimizer.state_dict(), lrate)
            print("Time per epoch : ", time.time() - start_time, " seconds")
        elif step == total_epoch - 1:
            end_model = os.path.join(model_save_path, "Conf_mod{:03d}-tc1000.pt".format(step + 1))
            print("Saving Ending Model")
            torch.save(model.state_dict(), end_model)
            end_lr = os.path.join(model_save_path, "Conf_lr{:03d}-tc1000.pt".format(step + 1))
            torch.save(optimizer.state_dict(), end_lr)
        else:
            worst_model = os.path.join(model_save_path, "Conf_mod{:03d}-tc1000.pt".format(step + 1))
            print("WORST: not saving: ", worst_model)


if __name__ == "__main__":
    print("Using GPU : ", torch.cuda.get_device_name())
    print("Using GPU Number: ", torch.cuda.current_device())
    run(model = model, total_epoch = total_epochs, best_loss = inf)
    print("closing the Writer...!!")
        
