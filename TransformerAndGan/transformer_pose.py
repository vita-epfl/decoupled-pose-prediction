
from torch import Tensor
import torch.nn.functional as f
import torch
from torch import nn
import numpy as np

import utils
## Model parameters
import transformer_network

        
num_layers = 1
d_model = 56 #2*28 
dff = 128 #64
num_heads = 4
dropout_rate = 0.20 # 0.2
torch.autograd.set_detect_anomaly(True)
transformer = transformer_network.Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    pe_input=1000, 
    pe_target=1000,
    rate=dropout_rate)

import args_module
args = args_module.args()
args.model_path = 'models/transformer_pose.pkl'
transformer.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)))

transformer = transformer.to(args.device)

import DataLoader
train = DataLoader.data_loader(args)
args.dtype = 'valid'
args.save_path = args.save_path.replace('train', 'val')
args.file = args.file.replace('train', 'val')
val = DataLoader.data_loader(args)

criterion = nn.MSELoss()
bce = nn.BCELoss()
#args.lr = 1.0
optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=240,
                                                 threshold = 5e-7, verbose=True)
best_model_val = 10000000

import time
for epoch in range(args.n_epochs):
    start = time.time()

    epoch_loss_p_train = 0.
    epoch_loss_v_train = 0.
    epoch_loss_train = 0.
    epoch_acc_train = 0. 
    vim_train = 0.
    fde_train = 0.
    ade_train = 0.



    epoch_loss_p_val = 0.
    epoch_loss_v_val = 0.
    epoch_loss_val = 0.
    epoch_acc_val = 0. 
    vim_val = 0.
    fde_val = 0.
    ade_val = 0.

    counter = 0
    Counter = 0
    transformer.train()
    for idx, (obs_kp, obs_skp, obs_frames, obs_masks, target_kp, target_skp, target_frames, future_masks) in enumerate(train):

        counter += 1
        obs_kp = obs_kp.to(device=args.device)
        obs_skp = obs_skp.to(device=args.device)
        target_kp = target_kp.to(device=args.device)
        target_skp = target_skp.to(device=args.device)
        obs_masks = obs_masks.to(device=args.device)
        future_masks = future_masks.to(device=args.device)
        
        ##############################################
        # using the transformer to predict pose
        inp = torch.cat((obs_kp[:,1:-1,:], obs_skp[:,:-1,:]), dim=-1)
        tar_inp = torch.cat((obs_kp[:,-1,:].unsqueeze(1), obs_skp[:,-1,:].unsqueeze(1)), dim=-1)
        tar_real = torch.cat((target_kp, target_skp), dim=-1)

        pose_vel_preds = torch.tensor([], device=args.device)
        predicted_masks = torch.tensor([], device=args.device)
        transformer.zero_grad()
        for t in range(args.output):

            pose_vel_pred, predicted_mask, _ = transformer(inp=inp, tar=tar_inp,
                                 training=True,
                                 enc_padding_mask=None,
                                 look_ahead_mask=None,
                                 dec_padding_mask=None)
            pose_vel_preds = torch.cat((pose_vel_preds, pose_vel_pred), dim=1)
            predicted_masks = torch.cat((predicted_masks, predicted_mask), dim=1)
            inp = torch.cat((inp, tar_inp), dim=1)
            tar_inp = pose_vel_pred

        pose_preds = pose_vel_preds[:,:,:28]

        ##############################################
        # defining the loss
        loss_v = bce(predicted_masks, future_masks) 
        loss_p = criterion(pose_vel_preds, tar_real)
        loss = loss_v + loss_p
        loss.backward()
        optimizer.step()

        epoch_loss_p_train += loss_p
        epoch_loss_v_train += loss_v
        epoch_loss_train += loss

        #############################################
        # calculating the accuracy of visibility masks

        predicted_masks = torch.where(predicted_masks>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))
        epoch_acc_train += sum(predicted_masks.view(-1)==future_masks.view(-1))
        #Counter += predicted_masks.view(-1).shape[0]
        Counter += predicted_masks.shape[0]
        vim_train += utils.myVIM(pose_preds, tar_real[:,:,:28], predicted_masks)
        fde_train += utils.FDE_keypoints(pose_preds, tar_real[:,:,:28])
        ade_train += utils.ADE_keypoints(pose_preds, tar_real[:,:,:28])

    epoch_loss_p_train /= counter 
    epoch_loss_v_train /= counter
    epoch_loss_train /= counter
    epoch_acc_train /= (Counter*args.output*14)
    vim_train /= (Counter*args.output)
    fde_train /= counter
    ade_train /= counter


    scheduler.step(loss)

    counter = 0
    Counter = 0
    transformer.eval()
    for idx, (obs_kp, obs_skp, obs_frames, obs_masks, target_kp, target_skp, target_frames, future_masks) in enumerate(val):

        counter += 1
        obs_kp = obs_kp.to(device=args.device)
        obs_skp = obs_skp.to(device=args.device)
        target_kp = target_kp.to(device=args.device)
        target_skp = target_skp.to(device=args.device)
        obs_masks = obs_masks.to(device=args.device)
        future_masks = future_masks.to(device=args.device)

        with torch.no_grad():
            
            inp = torch.cat((obs_kp[:,1:-1,:],obs_skp[:,:-1,:]), dim=-1)
            tar_inp = torch.cat((obs_kp[:,-1,:].unsqueeze(1), obs_skp[:,-1,:].unsqueeze(1)), dim=-1)
            tar_real = torch.cat((target_kp, target_skp), dim=-1)
            pose_vel_preds = torch.tensor([], device=args.device)
            predicted_masks = torch.tensor([], device=args.device)

            for t in range(args.output):
                 
                pose_vel_pred, predicted_mask, _ = transformer(inp=inp, tar=tar_inp,
                                 training=False, 
                                 enc_padding_mask=None, 
                                 look_ahead_mask=None, 
                                 dec_padding_mask=None)
                pose_vel_preds = torch.cat((pose_vel_preds, pose_vel_pred), dim=1)
                predicted_masks = torch.cat((predicted_masks, predicted_mask), dim=1)
                inp = torch.cat((inp, tar_inp),dim=1)
                tar_inp = pose_vel_pred
            pose_preds = pose_vel_preds[:,:,:28]


            ##############################################
            # defining the loss

            loss_v = bce(predicted_masks, future_masks) 
            loss_p = criterion(pose_vel_preds, tar_real)
            loss = loss_v + loss_p

            epoch_loss_p_val += loss_p
            epoch_loss_v_val += loss_v
            epoch_loss_val += loss

            #############################################
            # calculating the accuracy of visibility masks

            predicted_masks = torch.where(predicted_masks>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))
            epoch_acc_val += sum(predicted_masks.view(-1)==future_masks.view(-1))
            #Counter += predicted_masks.view(-1).shape[0]
            Counter += predicted_masks.shape[0]
            vim_val += utils.myVIM(pose_preds, tar_real[:,:,:28], predicted_masks)
            fde_val += utils.FDE_keypoints(pose_preds, tar_real[:,:,:28])
            ade_val += utils.ADE_keypoints(pose_preds, tar_real[:,:,:28])




    epoch_loss_p_val /= counter 
    epoch_loss_v_val /= counter
    epoch_loss_val /= counter
    epoch_acc_val /= (Counter*args.output*14)
    vim_val /= (Counter*args.output)
    fde_val /= counter
    ade_val /= counter


    print('e:', epoch,
         '<<< L2: %.6f'% epoch_loss_train,
         '| lp: %.6f'% epoch_loss_p_train,
         '| lv: %.6f'% epoch_loss_v_train,
         '| acc: %.6f'% epoch_acc_train,
         '| vim: %.6f'%vim_train,
         '| fde: %.6f'%fde_train,
         '| ade: %.6f'%ade_train,
         '>>> <<< L2: %.6f'% epoch_loss_val,
         '| lp: %.6f'% epoch_loss_p_val,
         '| lv: %.6f'% epoch_loss_v_val,
         '| acc: %.6f'% epoch_acc_val,
         '| vim: %.6f'%vim_val,
         '| fde: %.6f'%fde_val,
         '| ade: %.6f'%ade_val,
         '>>> t:%.4f'%(time.time()-start))
    #if(epoch % 10 == 0 and epoch != 0):
    #     torch.save(transformer.state_dict(), args.model_path)
    if epoch_loss_val < best_model_val:
         best_model_val = epoch_loss_val
         torch.save(transformer.state_dict(), args.model_path)

#torch.save(transformer.state_dict(), args.model_path)
    
