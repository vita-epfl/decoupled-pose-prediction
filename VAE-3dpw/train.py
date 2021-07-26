import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
#import cv2
import utils

from PIL import Image, ImageDraw
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import json
import viz
import args_module

args = args_module.args()
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def visu(doc, obs_p, pred_p, target_p, obs_f, target_f):

    #g = torch.Generator()
    #g.manual_seed(epoch)
    for i in range(10):
       #i = torch.randint(low=0, high=obs_p.shape[0]-1, size=(1,), generator=g).item()
       viewer = viz.Seq3DPose(window=1.0)
       prefix='result_'+doc+'/scenario_{:02d}/'.format(i)
       seq_in = obs_p[i]
       viewer.center(seq_in[..., 4:6, 0].mean(), seq_in[..., 4:6, 1].mean(), seq_in[..., 4:6, 2].mean())
       viewer.view(seq_in, 'Input', prefix=prefix)
       seq_ex = pred_p[i]
       viewer.view(seq_ex, 'Prediction', prefix=prefix)
       seq_exe = target_p[i]
       viewer.view(seq_exe, 'GroundTruth', prefix=prefix)
       print("deb: ", pred_p[i].shape, target_p[i].shape)
       vim = utils.myVIM(pred_p[i].view(2,14,39).permute(1,0,2), target_p[i].view(2,14,39).permute(1,0,2))
       with open('result_'+doc+'/scenario_{:02d}/'.format(i)+"info", 'w') as f:
           f.write("vim: {}\n".format(vim/2))
           lst = " ".join(obs_f[i])
           f.write("ObservationFrames: {}\n".format(lst))
           lst = " ".join(target_f[i])
           f.write("FutureFrames: {}\n".format(lst))


def mse(pred, true):
    seq_len, batch, l = pred.shape 
    #return torch.mean(torch.sqrt(torch.sum((pred-true)**2, dim=-1)/13))
    return torch.mean((pred-true)**2)

VISUALIZE = False #True
MODEL_VERSION = 'avg_hip' 
#MODEL_VERSION = 'neck' 
#MODEL_VERSION = 'rankle' #right ankle 

#mse = nn.MSELoss()
#l1e = nn.L1Loss()
#l1e = nn.MSELoss()
bce = nn.BCELoss()
train_s_scores = []
train_pose_scores=[]
val_pose_scores=[]
train_c_scores = []
val_s_scores   = []
val_c_scores   = []

import DataLoader
train = DataLoader.data_loader(args)
args.dtype = 'valid'
args.save_path = args.save_path.replace('train', 'val')
args.file = args.file.replace('train', 'val')
#val = DataLoader.data_loader(args)
val_all = DataLoader.data_loader_val(args)

import DataLoader_test
test = DataLoader_test.data_loader_test()

best_model_val = 100000.

import model_vae 
encoder = model_vae.Encoder(pose_dim=39, h_dim=32, latent_dim=16)
decoder = model_vae.Decoder(pose_dim=39, h_dim=32, latent_dim=16)
net = model_vae.VAE(Encoder=encoder, Decoder=decoder)
if torch.cuda.is_available():
    net.cuda()
net.double()

optimizer = optim.Adam( net.parameters(), lr=0.004)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=35, threshold = 1e-8, verbose=True)
coeff = 0.2
print('='*100)
print('Training ...')

for epoch in range(args.n_epochs):
    start = time.time()
    
    ade_train  = 0
    fde_train  = 0
    vim_train = 0
    loss_train = 0

    counter = 0
    net.train()
    for idx, (obs_p, obs_s, obs_f, target_p, target_s, target_f, start_end_idx) in enumerate(train):

        batch = obs_p.size(1)
        counter += batch 
        obs_p = obs_p.to(device='cpu').double()
        obs_s = obs_s.to(device='cpu').double()
        target_p = target_p.to(device='cpu').double()
        target_s = target_s.to(device='cpu').double()
        
        ###########################################################

        net.zero_grad()
        ########################################################################
        ######predicting the local speed using VAE and calculate loss########### 
        ########################################################################
        output, mean, log_var = net(obs_s.double())
        loss = model_vae.vae_loss_function(target_s.double(), output, mean, log_var) #- 0.0001* torch.norm(output)
        ########################################################################

        ###########################################################
        speed_preds = output
        preds_p = utils.speed2pos(speed_preds, obs_p) 
        ###########################################################

        #####################total loss############################
        #version 1: disentangled
        #loss = loss_g + coeff*loss_l #+ 0.0001*mse(speed_preds, target_s)
        #version 2: 
        #loss = mse(speed_preds, target_s)
        #########################debug#############################
        if(epoch%10 == 0): print(f"deb: loss {loss.item()} ")
        ####################backward and optimize##################
        loss.backward()
        ###########################################################
        optimizer.step()

        #####################calculating the metrics########################
        ade_train += float(utils.ADE_c(preds_p, target_p))
        fde_train += float(utils.FDE_c(preds_p, target_p))
        vim_train += utils.myVIM(preds_p, target_p)
        loss_train += loss.item()*batch
        ####################################################################

    loss_train /= counter
    ade_train  /= counter
    fde_train  /= counter    
    vim_train  /= counter    
    scheduler.step(loss_train)
 
    acc_val = 0
    ade_val  = 0
    fde_val  = 0
    vim_val = 0
    loss_val = 0

    counter = 0
    net.eval()
    for idx, (obs_p, obs_s, obs_f, target_p, target_s, target_f, start_end_idx) in enumerate(val_all):

        batch = obs_p.size(1) 
        counter += batch 
        obs_p = obs_p.to(device='cpu').double()
        obs_s = obs_s.to(device='cpu').double()
        target_p = target_p.to(device='cpu').double()
        target_s = target_s.to(device='cpu').double()
        
        with torch.no_grad():

            ########################################################################
            ######predicting the local speed using VAE and calculate loss########### 
            ########################################################################
            output, mean, log_var = net(obs_s.double())
            loss = model_vae.vae_loss_function(target_s.double(), output, mean, log_var)
            ########################################################################

            ###########################################################
            speed_preds = output
            preds_p = utils.speed2pos(speed_preds, obs_p) 
            ###########################################################

            #####################total loss############################
            #version 1: disentangled
            #loss = loss_g + coeff*loss_l
            #version 2:
            #loss = mse(speed_preds, target_s)
            ###########################################################
        
            #####################calculating the metrics########################
            ade_val += float(utils.ADE_c(preds_p, target_p))
            fde_val += float(utils.FDE_c(preds_p, target_p))
            vim_val += utils.myVIM(preds_p, target_p)
            loss_val += loss.item()*batch
            ####################################################################



        ########################write val predictions to file##########################
        obs_p_val = obs_p
        target_p_val = target_p
        preds_p_val = preds_p
        obs_f_val = obs_f
        target_f_val = target_f
        start_end_idx_val = start_end_idx
        ###############################################################################
 
    loss_val /= counter
    ade_val  /= counter
    fde_val  /= counter    
    vim_val  /= counter    

    print("e: %d "%epoch,
          "|loss_t: %0.6f "%loss_train,
          "|loss_v: %0.6f "%loss_val,
          "|fde_t: %0.6f "%fde_train,
          "|fde_v: %0.6f "%fde_val,
          "|ade_t: %0.6f "%ade_train, 
          "|ade_v: %0.6f "%ade_val, 
          "|vim_t: %0.6f "%vim_train,
          "|vim_v: %0.6f "%vim_val) 

    ########################write test predictions to file##########################
    if(vim_val < best_model_val):
        best_model_val = vim_val 

        print(f" \n saving the best model with vim_train {vim_train} and vim_val {vim_val} \n")
        ######################saving the checkpoints######################
        torch.save(net.state_dict(), 'checkpoint.pkl')

        net.eval()
        for idx, (obs_p, obs_s, obs_f, target_f, start_end_idx) in enumerate(test):

            obs_p = obs_p.to(device='cpu').double()
            obs_s = obs_s.to(device='cpu').double()
            
            batch = obs_p.size(1)

            with torch.no_grad():
                output, mean, log_var = net(obs_s.double())
                speed_preds = output
                preds_p = utils.speed2pos(speed_preds, obs_p) 

        #preds_p => (14, 170, 39) => alist => (85, 2, 14, 39) 
        alist = []
        for _, (start, end) in enumerate(start_end_idx):
            alist.append(preds_p.permute(1,0,2)[start:end].tolist())
        with open('3dpw_predictions.json', 'w') as f:
            f.write(json.dumps(alist))
        #(85 2 14 39)

        alist = []
        for _, (start, end) in enumerate(start_end_idx_val):
            alist.append(preds_p_val.permute(1,0,2)[start:end].tolist())
        with open('3dpw_predictions_val.json', 'w') as f:
            f.write(json.dumps(alist))

        #########################visualization of results##############################
        if(vim_val < 0.15):
            visu("VAE", obs_p_val.permute(1,0,2).view(-1, 2, 16, 13, 3), preds_p_val.permute(1,0,2).view(-1, 2, 14, 13, 3), target_p_val.permute(1,0,2).view(-1, 2, 14, 13, 3), obs_f_val, target_f_val)
        ###############################################################################

print('='*100) 
print('Saving ...')
print('Done !')
