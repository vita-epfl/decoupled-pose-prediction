
from torch import Tensor
import torch.nn.functional as f
import torch
from torch import nn
import numpy as np

import utils
## Model parameters
import transformer_network

class visibility_model(nn.Module):
    def __init__(self, args):
        super(visibility_model, self).__init__()

        self.encoder = nn.LSTM(input_size=28, hidden_size=args.hidden_size)
        self.linear = nn.Linear(28, 28)
        #self.linear = nn.Linear(args.hidden_size, 28)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pose):
       
        #poses => (n_batch, seq_len, 28) 
        #_, (hs, cs) = self.encoder(pose.permute(1,0,2))    # _ => (seq_len, n_batch, hidden)
        #return self.sigmoid(self.linear(_.permute(1,0,2)))  #=> (n_batch, seq, 28)

        #_, (hs, cs) = self.encoder(pose.permute(1,0,2))    # _ => (seq_len, n_batch, hidden)
        return self.sigmoid(self.linear(pose))  #=> (n_batch, seq, 28)

        
num_layers = 1
d_model = 56 #2*28 
dff = 256
num_heads = 4
dropout_rate = 0.0001

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
args.model_path = 'models/transformer_pose_vel.pkl'
model_path_net = 'models/transformer_pose_vel_net.pkl'
#transformer.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)))

transformer = transformer.to(args.device)

net = visibility_model(args)
#net.load_state_dict(torch.load(model_path_net, map_location=torch.device(args.device)))
net = net.to(args.device)

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
voptimizer = torch.optim.Adam(net.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=240,
                                                 threshold = 5e-7, verbose=True)

import time
for epoch in range(args.n_epochs):
    start = time.time()
    epoch_loss_train = 0.
    epoch_loss_visib_train = 0.
    epoch_acc_train = 0.
    epoch_count_train = 0
    
    vim_train = 0.
    fde_train = 0.
    ade_train = 0.

    vim_val = 0.
    fde_val = 0.
    ade_val = 0.
    epoch_loss_val = 0.
    epoch_loss_visib_val = 0.
    epoch_acc_val = 0.
    epoch_count_val = 0


    counter = 0
    transformer.train()
    for idx, (obs_kp, obs_skp, obs_frames, obs_masks, target_kp, target_skp, target_frames, future_masks) in enumerate(train):

        counter += 1
        obs_kp = obs_kp.to(device=args.device)
        obs_skp = obs_skp.to(device=args.device)
        target_kp = target_kp.to(device=args.device)
        target_skp = target_skp.to(device=args.device)
        obs_masks = obs_masks.to(device=args.device)
        future_masks = future_masks.to(device=args.device)
        
        obs_masks = torch.repeat_interleave(obs_masks, 2, dim=-1)
        future_masks = torch.repeat_interleave(future_masks, 2, dim=-1)

        ##############################################
        inp = torch.cat((obs_kp[:,:-1,:],obs_skp), dim=-1)
        tar_inp = torch.cat((obs_kp[:,-1,:].unsqueeze(1), obs_skp[:,-1,:].unsqueeze(1)), dim=-1)
        tar_real = torch.cat((target_kp, target_skp), dim=-1)

        pose_vel_preds = torch.tensor([], device=args.device)
        transformer.zero_grad()
        net.zero_grad()
        for t in range(args.output):

            pose_vel_pred, _ = transformer(inp=inp, tar=tar_inp,
                                 training=True,
                                 enc_padding_mask=None,
                                 look_ahead_mask=None,
                                 dec_padding_mask=None)
            pose_vel_preds = torch.cat((pose_vel_preds, pose_vel_pred), dim=1)
            inp = torch.cat((inp, tar_inp),dim=1)
            tar_inp = pose_vel_pred

        #pose_preds = utils.speed2bodypos(vel_preds, obs_skp)
        pose_preds = pose_vel_preds[:,:,:28]
        ##############################################
        predicted_masks = net(pose_preds.detach())
        visibility_loss = bce(predicted_masks, future_masks)

        epoch_loss_visib_train += visibility_loss
        loss = criterion(pose_vel_preds, tar_real)
        #loss = criterion(speed_preds, tar_real)
        loss.backward()
        optimizer.step()
        visibility_loss.backward()
        voptimizer.step()
        #pose_preds = utils.speed2bodypos(speed_preds, inp)

        epoch_loss_train += loss
        predicted_masks = torch.where(predicted_masks>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))
        epoch_acc_train += sum(predicted_masks.view(-1)==future_masks.view(-1))
        epoch_count_train += predicted_masks.view(-1).shape[0]
        #vim_train += utils.myVIM(pose_preds, tar_real[:,:,:28], future_masks)
        vim_train += utils.myVIM(pose_preds, tar_real[:,:,:28], predicted_masks)
        #vim_train += utils.VIM(tar_real[:,:,:28], pose_preds, "posetrack", predicted_masks)
        fde_train += utils.FDE_keypoints(pose_preds, tar_real[:,:,:28])
        ade_train += utils.ADE_keypoints(pose_preds, tar_real[:,:,:28])

    epoch_acc_train /= epoch_count_train
    epoch_loss_train /= counter
    epoch_loss_visib_train /= counter
    vim_train /= counter
    fde_train /= counter
    ade_train /= counter


    scheduler.step(loss)

    counter = 0
    transformer.eval()
    for idx, (obs_kp, obs_skp, obs_frames, obs_masks, target_kp, target_skp, target_frames, future_masks) in enumerate(val):

        counter += 1
        obs_kp = obs_kp.to(device=args.device)
        obs_skp = obs_skp.to(device=args.device)
        target_kp = target_kp.to(device=args.device)
        target_skp = target_skp.to(device=args.device)
        obs_masks = obs_masks.to(device=args.device)
        future_masks = future_masks.to(device=args.device)
        obs_masks = torch.repeat_interleave(obs_masks, 2, dim=-1)
        future_masks = torch.repeat_interleave(future_masks, 2, dim=-1)

        with torch.no_grad():
            
            inp = torch.cat((obs_kp[:,:-1,:],obs_skp), dim=-1)
            tar_inp = torch.cat((obs_kp[:,-1,:].unsqueeze(1), obs_skp[:,-1,:].unsqueeze(1)), dim=-1)
            tar_real = torch.cat((target_kp, target_skp), dim=-1)
            pose_vel_preds = torch.tensor([], device=args.device)

            for t in range(args.output):
                 
                pose_vel_pred, _ = transformer(inp=inp, tar=tar_inp, 
                                 training=True, 
                                 enc_padding_mask=None, 
                                 look_ahead_mask=None, 
                                 dec_padding_mask=None)
                pose_vel_preds = torch.cat((pose_vel_preds, pose_vel_pred), dim=1)
                inp = torch.cat((inp, tar_inp),dim=1)
                tar_inp = pose_vel_pred
            pose_preds = pose_vel_preds[:,:,:28]
            predicted_masks = net(pose_preds)
            visibility_loss = bce(predicted_masks, future_masks)
            epoch_loss_visib_val += visibility_loss
            #pose_preds = utils.speed2bodypos(vel_preds, obs_skp)
            loss = criterion(pose_vel_preds, tar_real)
            epoch_loss_val += loss

        #predicted_masks = torch.where(predicted_masks>=0.5, 1, 0)
        predicted_masks = torch.where(predicted_masks>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))
        epoch_acc_val += sum(predicted_masks.view(-1)==future_masks.view(-1))
        epoch_count_val += predicted_masks.view(-1).shape[0]
        #vim_val += utils.myVIM(pose_preds, tar_real[:,:,:28], future_masks)
        vim_val += utils.myVIM(pose_preds, tar_real[:,:,:28], predicted_masks)
        #vim_val += utils.VIM(tar_real[:,:,:28], pose_preds, "posetrack", predicted_masks)
        fde_val += utils.FDE_keypoints(pose_preds, tar_real[:,:,:28])
        ade_val += utils.ADE_keypoints(pose_preds, tar_real[:,:,:28])

    epoch_acc_val /= epoch_count_val
    epoch_loss_val /= counter
    epoch_loss_visib_val /= counter
    vim_val /= counter
    fde_val /= counter
    ade_val /= counter

                

    print('e:', epoch,
         '<<< L2: %.6f'% epoch_loss_train,
         '| vloss: %.6f'% epoch_loss_visib_train,
         '| acc: %.6f'% epoch_acc_train,
         '| vim: %.6f'%vim_train,
         '| fde: %.6f'%fde_train,
         '| ade: %.6f'%ade_train,
         '>>> <<< L2: %.6f'% epoch_loss_val,
         '| vloss: %.6f'% epoch_loss_visib_val,
         '| acc: %.6f'% epoch_acc_val,
         '| vim: %.6f'%vim_val,
         '| fde: %.6f'%fde_val,
         '| ade: %.6f'%ade_val,
         '>>> t:%.4f'%(time.time()-start))
    if(epoch % 10 == 0):
         pass
         #torch.save(transformer.state_dict(), args.model_path)
         #torch.save(net.state_dict(), model_path_net)

#torch.save(transformer.state_dict(), args.model_path)
#torch.save(net.state_dict(), model_path_net)
    
def main_transformer_pose_vel_output_vel():
    num_layers = 1
    d_model = 56 #2*28
    dff = 1024
    num_heads = 4
    dropout_rate = 0.0001
    
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
    args.model_path = 'models/transformer_pose_vel_output_vel.pkl'
 
    transformer.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)))
    transformer = transformer.to(args.device)
    
    
    import DataLoader
    train = DataLoader.data_loader(args)
    args.dtype = 'valid'
    args.save_path = args.save_path.replace('train', 'val')
    args.file = args.file.replace('train', 'val')
    val = DataLoader.data_loader(args)
    
    criterion = nn.MSELoss()
    #args.lr = 1.0
    optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1000,
                                                     threshold = 1e-7, verbose=True)
    
    import time
    for epoch in range(args.n_epochs):
        start = time.time()
        epoch_loss_train = 0.
        fde_train = 0.
        ade_train = 0.
    
        fde_val = 0.
        ade_val = 0.
        epoch_loss_val = 0.
    
    
        counter = 0
        transformer.train()
        for idx, (obs_kp, obs_skp, obs_frames, target_kp, target_skp, target_frames) in enumerate(train):
    
            counter += 1
            obs_kp = obs_kp.to(device=args.device)
            obs_skp = obs_skp.to(device=args.device)
            target_kp = target_kp.to(device=args.device)
            target_skp = target_skp.to(device=args.device)

            ##############################################
            #print(f' {obs_kp[:,:-1,:].shape} {obs_skp[:,:-1,:].shape}')
            inp = torch.cat((obs_kp[:,:-1,:],obs_skp), dim=-1)
            tar_inp = torch.cat((obs_kp[:,-1,:].unsqueeze(1), obs_skp[:,-1,:].unsqueeze(1)), dim=-1)
            tar_real = torch.cat((target_kp, target_skp), dim=-1)

            pose_vel_preds = torch.tensor([], device=args.device)
            transformer.zero_grad()
            for t in range(args.output):

                pose_vel_pred, _ = transformer(inp=inp, tar=tar_inp,
                                     training=True,
                                     enc_padding_mask=None,
                                     look_ahead_mask=None,
                                     dec_padding_mask=None)
                pose_vel_preds = torch.cat((pose_vel_preds, pose_vel_pred), dim=1)
                inp = torch.cat((inp, tar_inp),dim=1)
                tar_inp = pose_vel_pred

            #pose_preds = utils.speed2bodypos(vel_preds, obs_skp)
            pose_preds = utils.speed2bodypos(pose_vel_preds[:,:,28:], obs_kp) 

            ##############################################
            ###inp = obs_kp[:,:-1:]
            ###tar_inp = torch.cat((obs_kp[:,-1,:].unsqueeze(1), target_kp[:,:-1,:]), dim=1)
            ###tar_real = target_kp
            ##tar_inp = target_kp[:,:-1,:].to(device=args.device)
            ##tar_real = target_kp[:,1:,:].to(device=args.device)
            ##inp = obs_kp.to(device=args.device)
            #tar_inp = target_skp[:,:-1,:].to(device=args.device)
            #tar_real = target_skp[:,1:,:].to(device=args.device)
            #inp = obs_skp.to(device=args.device)
            #tar_inp_mask = transformer_network.create_look_ahead_mask(tar_inp.size(1)).to(device=args.device)
    
            #pose_preds, _ = transformer(inp=inp, tar=tar_inp, 
            #                     training=True, 
            #                     enc_padding_mask=None, 
            #                     look_ahead_mask=tar_inp_mask, 
            #                     dec_padding_mask=None)
            #speed_preds, _ = transformer(inp=inp, tar=tar_inp, 
            #                     training=True, 
            #                     enc_padding_mask=None, 
            #                     look_ahead_mask=tar_inp_mask, 
            #                     dec_padding_mask=None)
            loss = criterion(pose_vel_preds[:,:,28:], tar_real[:,:,28:])
            #loss = criterion(pose_vel_preds, tar_real)
            #loss = criterion(speed_preds, tar_real)
            loss.backward()
            optimizer.step()
            #pose_preds = utils.speed2bodypos(speed_preds, inp)
    
            epoch_loss_train += loss
            fde_train += utils.FDE_keypoints(pose_preds, tar_real[:,:,:28])
            ade_train += utils.ADE_keypoints(pose_preds, tar_real[:,:,:28])
    
        epoch_loss_train /= counter
        fde_train /= counter
        ade_train /= counter
    
    
        scheduler.step(loss)
    
        counter = 0
        transformer.eval()
        for idx, (obs_kp, obs_skp, obs_frames, target_kp, target_skp, target_frames) in enumerate(train):
    
            counter += 1
            obs_kp = obs_kp.to(device=args.device)
            obs_skp = obs_skp.to(device=args.device)
            target_kp = target_kp.to(device=args.device)
            target_skp = target_skp.to(device=args.device)
    
            with torch.no_grad():
                
                inp = torch.cat((obs_kp[:,:-1,:],obs_skp), dim=-1)
                tar_inp = torch.cat((obs_kp[:,-1,:].unsqueeze(1), obs_skp[:,-1,:].unsqueeze(1)), dim=-1)
                tar_real = torch.cat((target_kp, target_skp), dim=-1)
                pose_vel_preds = torch.tensor([], device=args.device)

                for t in range(args.output):
                     
                    pose_vel_pred, _ = transformer(inp=inp, tar=tar_inp, 
                                     training=True, 
                                     enc_padding_mask=None, 
                                     look_ahead_mask=None, 
                                     dec_padding_mask=None)
                    pose_vel_preds = torch.cat((pose_vel_preds, pose_vel_pred), dim=1)
                    inp = torch.cat((inp, tar_inp),dim=1)
                    tar_inp = pose_vel_pred
                pose_preds = utils.speed2bodypos(pose_vel_preds[:,:,28:], obs_kp)
                loss = criterion(pose_vel_preds[:,:,28:], tar_real[:,:,28:])
                epoch_loss_val += loss
            fde_val += utils.FDE_keypoints(pose_preds, tar_real[:,:,:28])
            ade_val += utils.ADE_keypoints(pose_preds, tar_real[:,:,:28])
    
        epoch_loss_val /= counter
        fde_val /= counter
        ade_val /= counter
    
                    
    
        print('e:', epoch,
             '<<< L2: %.6f'% epoch_loss_train,
             '| fde: %.6f'%fde_train,
             '| ade: %.6f'%ade_train,
             '>>> <<< L2: %.6f'% epoch_loss_val,
             '| fde: %.6f'%fde_val,
             '| ade: %.6f'%ade_val,
             '>>> t:%.4f'%(time.time()-start))
        if(epoch % 10 == 0):
             torch.save(transformer.state_dict(), args.model_path)

    torch.save(transformer.state_dict(), args.model_path)
    
