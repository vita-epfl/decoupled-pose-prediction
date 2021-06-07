
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
        self.linear = nn.Linear(args.hidden_size, 14)
        #self.linear = nn.Linear(args.hidden_size, 28)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pose):
       
        #poses => (n_batch, seq_len, 28) 
        out, (hs, cs) = self.encoder(pose.permute(1,0,2))    # _ => (seq_len, n_batch, hidden)
        out = out.permute(1,0,2)  #out => n_batch, seq_len, 28
        out = self.linear(out) #out => n_batch, seq_len, 14
        out = self.sigmoid(out) #out => n_batch, seq_len, 14
        return out 

        ##_, (hs, cs) = self.encoder(pose.permute(1,0,2))    # _ => (seq_len, n_batch, hidden)
        ##return self.sigmoid(self.linear(pose))  #=> (n_batch, seq, 28)

        
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
args.model_path = 'models/transformer_pose_vel_alldata.pkl'
model_path_net = 'models/transformer_pose_vel_net_alldata.pkl'
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
    n_batches = 0 


    counter = 0
    transformer.train()
    for idx, ((obs_kp, obs_skp, obs_frames, obs_masks, target_kp, target_skp, target_frames, future_masks), 
              (obs_kp_v, obs_skp_v, obs_frames_v, obs_masks_v, target_kp_v, target_skp_v, target_frames_v, future_masks_v)
              ) in enumerate(zip(train,val)):

        obs_kp = torch.cat((obs_kp, obs_kp_v), dim=0)
        obs_skp = torch.cat((obs_skp, obs_skp_v), dim=0)
        obs_frames = obs_frames + obs_frames_v #torch.cat((obs_frames, obs_frames_v), dim=0)
        obs_masks = torch.cat((obs_masks, obs_masks_v), dim=0)
        target_kp = torch.cat((target_kp, target_kp_v), dim=0)
        target_skp = torch.cat((target_skp, target_skp_v), dim=0)
        target_frames = target_frames + target_frames_v #torch.cat((target_frames, target_frames_v), dim=0)
        future_masks = torch.cat((future_masks, future_masks_v), dim=0)

        counter += 1
        obs_kp = obs_kp.to(device=args.device)
        obs_skp = obs_skp.to(device=args.device)
        target_kp = target_kp.to(device=args.device)
        target_skp = target_skp.to(device=args.device)
        obs_masks = obs_masks.to(device=args.device)
        future_masks = future_masks.to(device=args.device)
        
        #obs_masks = torch.repeat_interleave(obs_masks, 2, dim=-1)
        #future_masks = torch.repeat_interleave(future_masks, 2, dim=-1)

        ##############################################
        inp = torch.cat((obs_kp[:,1:-1,:], obs_skp[:,:-1,:]), dim=-1)
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
        loss = criterion(pose_vel_preds[:,:,:28], tar_real[:,:,:28])
        loss.backward()
        optimizer.step()
        ##############################################
        predicted_masks = net(pose_preds.detach())
        visibility_loss = bce(predicted_masks, future_masks)
        epoch_loss_visib_train += visibility_loss
        #loss = criterion(speed_preds, tar_real)
        visibility_loss.backward()
        voptimizer.step()
        #pose_preds = utils.speed2bodypos(speed_preds, inp)

        epoch_loss_train += loss
        predicted_masks = torch.where(predicted_masks>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))
        epoch_acc_train += sum(predicted_masks.view(-1)==future_masks.view(-1))
        epoch_count_train += predicted_masks.view(-1).shape[0]
        n_batches += predicted_masks.shape[0]
        #vim_train += utils.myVIM(pose_preds, tar_real[:,:,:28], future_masks)
        vim_train += utils.myVIM(pose_preds, tar_real[:,:,:28], predicted_masks)
        #vim_train += utils.VIM(tar_real[:,:,:28], pose_preds, "posetrack", predicted_masks)
        fde_train += utils.FDE_keypoints(pose_preds, tar_real[:,:,:28])
        ade_train += utils.ADE_keypoints(pose_preds, tar_real[:,:,:28])

    epoch_acc_train /= epoch_count_train
    epoch_loss_train /= counter
    epoch_loss_visib_train /= counter
    vim_train /= (n_batches*args.output)
    fde_train /= counter
    ade_train /= counter


    scheduler.step(loss)

    print('e:', epoch,
         '<<< L2: %.6f'% epoch_loss_train,
         '| vloss: %.6f'% epoch_loss_visib_train,
         '| acc: %.6f'% epoch_acc_train,
         '| vim: %.6f'%vim_train,
         '| fde: %.6f'%fde_train,
         '| ade: %.6f'%ade_train,
         '>>> t:%.4f'%(time.time()-start))
    if(epoch % 10 == 0 and epoch != 0):
         torch.save(transformer.state_dict(), args.model_path)
         torch.save(net.state_dict(), model_path_net)

torch.save(transformer.state_dict(), args.model_path)
torch.save(net.state_dict(), model_path_net)
    
