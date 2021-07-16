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

from PIL import Image, ImageDraw
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import args_module
import random
import json

import args_module
args = args_module.args()

mse = nn.MSELoss()
#l1e = nn.L1Loss()
l1e = nn.MSELoss()
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
optimizer = optim.Adam(net.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, 
                                                 threshold = 1e-8, verbose=True)
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
    for idx, (obs_p, obs_s, obs_a, obs_f, target_p, target_s, target_a, target_f, start_end_idx) in enumerate(train):

        counter += 1        
        obs_p = obs_p.to(device='cuda')
        obs_s = obs_s.to(device='cuda')
        obs_a = obs_a.to(device='cuda')
        target_p = target_p.to(device='cuda')
        target_s = target_s.to(device='cuda')
        target_a = target_a.to(device='cuda')
        
        net.zero_grad()
        speed_preds, preds_p = net(obs_pose=obs_p, obs_speed=obs_s, start_end_idx=start_end_idx)

        #predicted_a = torch.cat(( (speed_preds[0]-obs_s[-1]).unsqueeze(0), speed_preds[1:]-speed_preds[:-1]), dim=0)
        #loss = l1e(speed_preds, target_s)
        loss = l1e(preds_p, target_p)
    
        #preds_p = utils.speed2pos(0.5*predicted_a + speed_preds, obs_p) 
        ade_train += float(utils.ADE_c(preds_p, target_p))
        fde_train += float(utils.FDE_c(preds_p, target_p))
        
        #print(f" loss_speed {0.7*speed_loss} loss_mask {0.3*mask_loss} ")
        loss.backward()

        optimizer.step()
        
        vim_train += utils.myVIM(preds_p, target_p)

        loss_train += loss  

    scheduler.step(loss)
    loss_train /= counter
    ade_train  /= counter
    fde_train  /= counter    
    vim_train  /= counter    
 
    counter = 0
    net.eval()
    for idx, (obs_p, obs_s, obs_a, obs_f, target_p, target_s, target_a, target_f, start_end_idx) in enumerate(val):

        counter += 1        
        obs_p = obs_p.to(device='cuda')
        obs_s = obs_s.to(device='cuda')
        obs_a = obs_a.to(device='cuda')
        target_p = target_p.to(device='cuda')
        target_s = target_s.to(device='cuda')
        target_a = target_a.to(device='cuda')
        
        with torch.no_grad():
            speed_preds, preds_p = net(obs_pose=obs_p, obs_speed=obs_s, start_end_idx=start_end_idx)
            #loss  = l1e(speed_preds, target_s)
            loss  = l1e(preds_p, target_p)
    
            #predicted_a = torch.cat(( (speed_preds[0]-obs_s[-1]).unsqueeze(0), speed_preds[1:]-speed_preds[:-1]), dim=0)

            #preds_p = utils.speed2pos(0.5*predicted_a + speed_preds, obs_p) 
            ade_val += float(utils.ADE_c(preds_p, target_p))
            fde_val += float(utils.FDE_c(preds_p, target_p))
        
            vim_val += utils.myVIM(preds_p, target_p)
            loss_val += loss  

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

    if(False and vim_train < best_model_val):
        best_model_val = vim_train 
        torch.save(net.state_dict(), 'checkpoint.pkl')
        import DataLoader_test
        test = DataLoader_test.data_loader_test()
        net.eval()
        for idx, (obs_p, obs_s, obs_f, target_f, start_end_idx) in enumerate(test):

            obs_p = obs_p.to(device='cuda')
            obs_s = obs_s.to(device='cuda')
            
            with torch.no_grad():
                speed_preds =  net(obs_pose=obs_p, obs_speed=obs_s, start_end_idx=start_end_idx)
                preds_p = utils.speed2pos(speed_preds, obs_p) 

        #preds_p => (14, 170, 39) => alist => (85, 2, 14, 39) 
        alist = []
        for _, (start, end) in enumerate(start_end_idx):
            alist.append(preds_p.permute(1,0,2)[start:end].tolist())
        with open('3dpw_predictions.json', 'w') as f:
            f.write(json.dumps(alist))
        #(85 2 14 39)

print('='*100) 
print('Saving ...')
print('Done !')
