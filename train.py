#import os
import torch 
#import torchvision
import torch.nn as nn
#import torchvision.transforms as transforms
import torch.optim as optim
#import matplotlib.pyplot as plt
#import torch.nn.functional as F
 
#from torchvision import datasets
#from torchvision.utils import save_image
import numpy as np
#from torch.utils.data import Dataset, DataLoader
#import numpy as np
#import glob
import time
#import cv2
import utils
import json

from PIL import Image, ImageDraw
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import args_module
import random

import args_module
args = args_module.args()
scale = args.scale
scalex = args.scalex
scaley = args.scaley

mse = nn.MSELoss()

l1e = nn.L1Loss()
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
val = DataLoader.data_loader(args)

best_model_val = 100000.

#import network
#net = network.LSTMwithPooling().to(args.device)
import model 
net = model.LSTM().to(args.device)
#net.load_state_dict(torch.load('checkpoint_posetrack.pkl'))
optimizer = optim.Adam(net.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=45, 
                                                 threshold = 1e-5, verbose=True)
print('='*100)
print('Training ...')

for epoch in range(args.n_epochs):
    start = time.time()
    
    acc_train = 0
    acc_val = 0
    ade_val  = 0
    fde_val  = 0
    vim_val = 0
    loss_val = 0
    ade_train  = 0
    fde_train  = 0
    vim_train = 0
    loss_train = 0

    counter = 0
    net.train()
    for idx, (obs_p, obs_s, obs_f, obs_m, target_p, target_s, target_f, target_m, start_end_idx, vo_m, vt_m) in enumerate(train):

        #print("deb: ", len(obs_f), len(obs_f[0]), obs_f[0])
        #print(len(obs_f), len(obs_f[0]))
        if(scale):
            seq_len, batch, l = obs_p.shape
            obs_p = obs_p.view(seq_len, batch, l//2, 2)
            obs_s = obs_s.view(-1, batch, l//2, 2)
            target_p = target_p.view(-1, batch, l//2, 2)
            target_s = target_s.view(-1, batch, l//2, 2)

            obs_p[:,:,:,0] /= scalex
            obs_p[:,:,:,1] /= scaley
            obs_s[:,:,:,0] /= scalex
            obs_s[:,:,:,1] /= scaley
            target_p[:,:,:,0] /= scalex
            target_p[:,:,:,1] /= scaley
            target_s[:,:,:,0] /= scalex
            target_s[:,:,:,1] /= scaley

            obs_p = obs_p.view(seq_len, batch, l)
            obs_s = obs_s.view(-1, batch, l)
            target_p = target_p.view(-1, batch, l)
            target_s = target_s.view(-1, batch, l)

        counter += 1        
        obs_p = obs_p.to(device='cuda')
        obs_s = obs_s.to(device='cuda')
        target_p = target_p.to(device='cuda')
        target_s = target_s.to(device='cuda')
        obs_m = obs_m.to(device='cuda')
        vt_m = vt_m.to(device='cuda')
        target_m = target_m.to(device='cuda')
        
        net.zero_grad()
        speed_preds, mask_preds =  net(obs_pose=obs_p, obs_speed=obs_s, obs_mask=obs_m, start_end_idx=start_end_idx)

        #speed_loss  = l1e(speed_preds, target_s) 
        speed_loss  = l1e(speed_preds*torch.repeat_interleave(target_m, 2, dim=-1), target_s) 
        #print(mask_preds.shape, mask_preds, obs_m.shape, obs_m)
        mask_loss  = bce(mask_preds, target_m)
    
        preds_p = utils.speed2pos(speed_preds, obs_p) 
        ade_train += float(utils.ADE_c(preds_p, target_p))
        fde_train += float(utils.FDE_c(preds_p, target_p))
        
        #print(f" loss_speed {speed_loss} loss_mask {mask_loss} ")
        loss= speed_loss + mask_loss
        loss.backward()

        optimizer.step()
        
        predicted_masks = torch.where(mask_preds>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))
        vim_train += utils.myVIM(preds_p, target_p, predicted_masks)

        #mask => (seq_len, batch, 14)
        acc_train += torch.mean((predicted_masks.view(-1,14) == target_m.view(-1,14)).type(torch.float))
  
        loss_train += loss  

    scheduler.step(loss)
    loss_train /= counter
    ade_train  /= counter
    fde_train  /= counter    
    vim_train  /= counter    
    acc_train /= counter
 
    counter = 0
    net.eval()
    for idx, (obs_p, obs_s, obs_f, obs_m, target_p, target_s, target_f, target_m, start_end_idx, vo_m, vt_m) in enumerate(val):

        if(scale):
            seq_len, batch, l = obs_p.shape
            obs_p = obs_p.view(seq_len, batch, l//2, 2)
            obs_s = obs_s.view(-1, batch, l//2, 2)
            target_p = target_p.view(-1, batch, l//2, 2)
            target_s = target_s.view(-1, batch, l//2, 2)

            obs_p[:,:,:,0] /= scalex
            obs_p[:,:,:,1] /= scaley
            obs_s[:,:,:,0] /= scalex
            obs_s[:,:,:,1] /= scaley
            target_p[:,:,:,0] /= scalex
            target_p[:,:,:,1] /= scaley
            target_s[:,:,:,0] /= scalex
            target_s[:,:,:,1] /= scaley

            obs_p = obs_p.view(seq_len, batch, l)
            obs_s = obs_s.view(-1, batch, l)
            target_p = target_p.view(-1, batch, l)
            target_s = target_s.view(-1, batch, l)

        counter += 1        
        obs_p = obs_p.to(device='cuda')
        obs_s = obs_s.to(device='cuda')
        target_p = target_p.to(device='cuda')
        target_s = target_s.to(device='cuda')
        obs_m = obs_m.to(device='cuda')
        target_m = target_m.to(device='cuda')
        vt_m = vt_m.to(device='cuda')
        
        with torch.no_grad():
            speed_preds, mask_preds =  net(obs_pose=obs_p, obs_speed=obs_s, obs_mask=obs_m, start_end_idx=start_end_idx)
            #(speed_preds, mask_preds) = net(obs_pose=obs_p, obs_pose_rel=obs_s, obs_mask=obs_m, seq_start_end=start_end_idx)
            speed_loss  = l1e(speed_preds, target_s)
            mask_loss  = bce(mask_preds, target_m)
            preds_p = utils.speed2pos(speed_preds, obs_p) 
            ade_val += float(utils.ADE_c(preds_p, target_p))
            fde_val += float(utils.FDE_c(preds_p, target_p))
            
            loss = speed_loss + mask_loss
            
            predicted_masks = torch.where(mask_preds>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))
            vim_val += utils.myVIM(preds_p, target_p, predicted_masks)
  
            loss_val += loss  
            acc_val += torch.mean((predicted_masks == target_m).type(torch.float))

    loss_val /= counter
    ade_val  /= counter
    fde_val  /= counter    
    vim_val  /= counter    
    acc_val /= counter
 
     
    #g = torch.Generator()
    #g.manual_seed(epoch%10)
    #rnd = torch.randint(low=0, high=obs_p.shape[0]-1, size=(1,), generator=g).item()
    #plotter(obs_frames, obs_pose, obs_mask, target_frames, preds_p, predicted_masks, target_pose, target_mask, epoch, idx, rnd, width=10)

    rnd = torch.randint(low=0, high=obs_p.shape[0]-1, size=(1,)).item()
    #utils.plotter(obs_f, obs_p, obs_m, target_f, preds_p, predicted_masks, target_p, target_m, epoch, idx, rnd, width=10)

    print("e: %d "%epoch,
          "|loss_t: %0.4f "%loss_train,
          "|loss_v: %0.4f "%loss_val,
          "|fde_t: %0.4f "%fde_train,
          "|fde_v: %0.4f "%fde_val,
          "|ade_t: %0.4f "%ade_train, 
          "|ade_v: %0.4f "%ade_val, 
          "|vim_t: %0.4f "%vim_train,
          "|vim_v: %0.4f "%vim_val, 
          "|acc_t: %0.4f "%acc_train, 
          "|acc_v: %0.4f "%acc_val)

    if(vim_train < best_model_val):
        best_model_val = vim_train
        torch.save(net.state_dict(), 'checkpoint_posetrack.pkl')
        import DataLoader_test
        test = DataLoader_test.data_loader_test()
        net.eval()
        for idx, (obs_p, obs_s, obs_f, obs_m, target_f, start_end_idx, vo_m) in enumerate(test):

            obs_p = obs_p.to(device='cuda')
            obs_s = obs_s.to(device='cuda')
            obs_m = obs_m.to(device='cuda')

            with torch.no_grad():
                speed_preds, mask_preds = net(obs_pose=obs_p, obs_speed=obs_s, obs_mask=obs_m, start_end_idx=start_end_idx)
                preds_p = utils.speed2pos(speed_preds, obs_p) 
                preds_m = torch.where(mask_preds>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))

        alist_p = []
        alist_m = []
        for _, (start, end) in enumerate(start_end_idx):
            alist_p.append(preds_p.permute(1,0,2)[start:end].tolist())
            alist_m.append(preds_m.permute(1,0,2)[start:end].tolist())

        with open('posetrack_predictions.json', 'w') as f:
            f.write(json.dumps(alist_p))
        with open('posetrack_masks.json', 'w') as f:
            f.write(json.dumps(alist_m))




print('='*100) 
print('Saving ...')
print('Done !')
