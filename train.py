import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import time

from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score

import DataLoader
from PIL import Image, ImageDraw
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import utils


import args_module
args = args_module.args()

import network
net = network.LSTM(args).to(args.device)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

train = DataLoader.data_loader(args)
args.dtype = 'valid'
args.save_path = args.save_path.replace('train', 'val')
args.file = args.file.replace('train', 'val')
val = DataLoader.data_loader(args)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, threshold = 1e-7, verbose=True)
#mse = nn.MSELoss()
mse = nn.L1Loss()
bce = nn.BCELoss()

best_model_val = 100000.
print('='*100)
print('Training ...')
for epoch in range(args.n_epochs):
    start = time.time()
    
    loss_train = 0.
    epoch_acc_train = 0. 
    fde_train = 0.
    ade_train = 0. 
    vim_train = 0. 
    
    counter = 0
    Counter = 0

    for idx, (obs_pose, obs_speed, obs_frames, obs_masks, \
              target_pose, target_speed, target_frames, future_masks) in enumerate(train):

        counter += 1
        obs_pose = obs_pose.to(device=args.device)
        obs_speed = obs_speed.to(device=args.device)

        target_pose = target_pose.to(device=args.device)
        target_speed = target_speed.to(device=args.device)

        obs_masks = obs_masks.to(device=args.device)
        future_masks = future_masks.to(device=args.device)

        optimizer.zero_grad()
        speed_preds, masks_preds = net(pose=obs_pose, vel=obs_speed, mask=obs_masks)
        mse_loss = mse(speed_preds, target_speed)
        bce_loss = bce(masks_preds, future_masks)
        pose_preds = utils.speed2bodypose(speed_preds, obs_pose)
        loss = mse_loss + .5*bce_loss
        loss.backward()
        optimizer.step()
        loss_train += loss 

        fde_train += utils.FDE_keypoints(pose_preds, target_pose)
        ade_train += utils.ADE_keypoints(pose_preds, target_pose)
       
        predicted_masks = torch.where(masks_preds>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))
        epoch_acc_train += sum(predicted_masks.view(-1)==future_masks.view(-1))
        Counter += predicted_masks.shape[0]
        vim_train += utils.myVIM(pose_preds, target_pose, predicted_masks)

    scheduler.step(loss)
    
    loss_train /= counter
    fde_train /= counter
    ade_train /= counter
    vim_train /= counter
    epoch_acc_train /= (Counter*args.output*14)

    loss_val = 0.
    epoch_acc_val = 0.
    fde_val = 0.
    ade_val = 0.
    vim_val = 0.

    counter = 0
    Counter = 0

    for idx, (obs_pose, obs_speed, obs_frames, obs_masks, \
              target_pose, target_speed, target_frames, future_masks) in enumerate(val):

        counter += 1
        obs_pose = obs_pose.to(device=args.device)
        obs_speed = obs_speed.to(device=args.device)

        target_pose = target_pose.to(device=args.device)
        target_speed = target_speed.to(device=args.device)

        obs_masks = obs_masks.to(device=args.device)
        future_masks = future_masks.to(device=args.device)

        with torch.no_grad():

            speed_preds, masks_preds  = net(pose=obs_pose, vel=obs_speed, mask=obs_masks)
            mse_loss = mse(speed_preds, target_speed)
            bce_loss = bce(masks_preds, future_masks)
            pose_preds = utils.speed2bodypose(speed_preds, obs_pose)
            loss = mse_loss + 1.*bce_loss
            loss_val += loss 

            fde_val += utils.FDE_keypoints(pose_preds, target_pose)
            ade_val += utils.ADE_keypoints(pose_preds, target_pose)

            predicted_masks = torch.where(masks_preds>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))
            epoch_acc_val += sum(predicted_masks.view(-1)==future_masks.view(-1))
            Counter += predicted_masks.shape[0]
            vim_val += utils.myVIM(pose_preds, target_pose, predicted_masks)

    #setting random seed to visualize results and compare between different methods
    g = torch.Generator()
    g.manual_seed(epoch%10)
    rnd = torch.randint(low=0, high=obs_pose.shape[0]-1, size=(1,), generator=g).item()
    utils.plotter(obs_frames, obs_pose, obs_masks, target_frames, pose_preds, predicted_masks, target_pose, future_masks, epoch, idx, rnd, width=10)


  
    loss_val /= counter
    fde_val /= counter
    ade_val /= counter
    vim_val /= counter 
    epoch_acc_val /= (Counter*args.output*14)

    print("e: %d "%epoch,
          "|loss_train: %0.4f "%loss_train, 
          "|loss_val: %0.4f "%loss_val, 
          "|acc_train: %0.4f "%epoch_acc_train, 
          "|acc: %0.4f "%epoch_acc_val, 
          "|vim_train: %0.4f "%vim_train, 
          "|vim: %0.4f "%vim_val, 
          "|fde_train: %0.4f "%fde_train, 
          "|fde_val: %0.4f "%fde_val, 
          "|ade_train: %0.4f "%ade_train,
          "|ade_val: %0.4f "%ade_val) 

    #Save the weights and results for submission to the challenge
    if( vim_train < best_model_val):
         best_model_val = vim_train
         torch.save(net.state_dict(), args.model_path)
         ################### generate predictions ##############
         # load test data
         import DataLoader_test
         args.dtype = 'test'
         test, info = DataLoader_test.data_loader(args)
         obs_kp, obs_skp, obs_frames, obs_masks, future_frames = next(iter(test)) 
         # move data to device
         obs_kp = obs_kp.to(device=args.device)
         obs_skp = obs_skp.to(device=args.device)
         obs_masks = obs_masks.to(device=args.device)

         with torch.no_grad():
             speed_preds,mask_preds = net(pose=obs_kp,vel=obs_skp,mask=obs_masks)
             pose_preds = utils.speed2bodypose(speed_preds, obs_kp)

         predicted_masks = torch.where(mask_preds>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))
         if(args.scale):
             n_batch, seq_len, l = pose_preds.shape
             assert(l == 28)
             pose_preds = pose_preds.view(n_batch, seq_len, l//2, 2)
             pose_preds[:,:,:,0] *= args.scalex 
             pose_preds[:,:,:,1] *= args.scaley
             pose_preds = pose_preds.view(n_batch, seq_len, l)

         # collect pedestrians
         lst = []
         lst_ = []
         count = -1
         for v_idx in range(len(info)):
            lst.append([])
            lst_.append([])
            for p_idx in range(info[v_idx]):
                count += 1
                lst[v_idx].append( pose_preds[count].tolist() )
                lst_[v_idx].append( predicted_masks[count].tolist() )

         # write it to file
         import json
         alst = json.dumps(lst)
         alst_ = json.dumps(lst_)
         with open('submission/posetrack_predictions.json', 'w') as f:
              f.write(alst)
         with open('submission/posetrack_masks.json', 'w') as f:
              f.write(alst_)


print('='*100) 
print('Done !')

    


