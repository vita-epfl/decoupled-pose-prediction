import torch
import torchvision
import torchvision.transforms.functional as TF
import pandas as pd
from ast import literal_eval
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import time
import json
from scipy import interpolate

class myJAAD(torch.utils.data.Dataset):
    def __init__(self, args):
        print('Loading', args.dtype, 'data ...')
        full_path = "/home/parsaeif/posetrack_challenge/somof_data_posetrack/"

        self.args = args
        #image_size: 1280, 720
        with open(os.path.join(full_path,"posetrack_"+args.dtype+"_frames_in.json"), 'r') as f:
            self.frames_in = json.load(f)
        #(306, 16)
        with open(os.path.join(full_path,"posetrack_"+args.dtype+"_masks_in.json"), 'r') as f:
            self.masks_in = json.load(f)

        with open(os.path.join(full_path,"posetrack_"+args.dtype+"_in.json"), 'r') as f:
            self.data_in = json.load(f)

        frames_out = []
        for i in range(len(self.frames_in)):
           frames_out.append([])
           path = "/".join(self.frames_in[i][-1].split("/")[:-1])
           last = int(self.frames_in[i][-1].split("/")[-1].split(".")[0]) + 1
           for j in range(14):
              frames_out[i].append(path+"/{:06d}.jpg".format(last+j))
        self.frames_out = frames_out

        with open(os.path.join(full_path,"posetrack_"+args.dtype+"_masks_out.json"), 'r') as f:
            self.masks_out = json.load(f)

        with open(os.path.join(full_path,"posetrack_"+args.dtype+"_out.json"), 'r') as f:
            self.data_out = json.load(f)
        
            

    def __len__(self):
        return len(self.data_in) 
    
    
    def __getitem__(self, idx):


        #preprocessing
        obs_p = torch.tensor(self.data_in[idx])
        obs_m = torch.tensor(self.masks_in[idx])
        obs_p[ torch.repeat_interleave(obs_m, 2, dim=-1) == 0 ] = np.NaN 
        obs_s = obs_p[:,1:,:] - obs_p[:,:-1,:]
        n_ped, seq_len, l = obs_s.shape
        _ = torch.ones(n_ped, seq_len, l//2)
        _[torch.isnan(obs_s[:,:,::2])] = 0
        velocity_obs_m = _

        target_p = torch.tensor(self.data_out[idx])
        target_m = torch.tensor(self.masks_out[idx])
        target_p[ torch.repeat_interleave(target_m, 2, dim=-1) == 0 ] = np.NaN
        target_s = torch.cat(((target_p[:,0,:]-obs_p[:,-1,:]).unsqueeze(1), target_p[:,1:,:]-target_p[:,:-1,:]), dim=1) 
        n_ped, seq_len, l = target_s.shape
        _ = torch.ones(n_ped, seq_len, l//2)
        _[torch.isnan(target_s[:,:,::2])] = 0
        velocity_target_m = _
        
        obs_s[torch.isnan(obs_s)] = 0
        obs_p[torch.isnan(obs_p)] = 0
        target_s[torch.isnan(target_s)] = 0
        target_p[torch.isnan(target_p)] = 0
        ###################

        return obs_p, obs_s, self.frames_in[idx], obs_m, target_p, target_s, self.frames_out[idx], target_m, velocity_obs_m, velocity_target_m

#    def __getitem__(self, idx):
#
#        obs_keypoints = torch.tensor(self.data_in[idx])   #n_ped,16,28
#        future_keypoints = torch.tensor(self.data_out[idx]) #n_ped, 14, 28
#
#        obs_speed_keypoints = (obs_keypoints[:,1:,:] - obs_keypoints[:,:-1,:]) #n_ped, 15, 28
#        true_speed = torch.cat(((future_keypoints[:,0,:]-obs_keypoints[:,-1,:]).unsqueeze(1), future_keypoints[:,1:,:]-future_keypoints[:,:-1,:]),dim=1)
#
#        return obs_keypoints, obs_speed_keypoints, self.frames_in[idx], torch.tensor(self.masks_in[idx]), \
#               future_keypoints, true_speed, self.frames_out[idx], torch.tensor(self.masks_out[idx])

def my_collate(batch):

    (obs_p, obs_s, obs_f, obs_m, target_p, target_s, target_f, target_m, v_obs_m, v_target_m) = zip(*batch)
    _len = [len(seq) for seq in obs_p]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx[:-1], cum_start_idx[1:])]

    obs_ff = []
    target_ff = []
    for i in range(len(_len)):
       for j in range(_len[i]):
          obs_ff.append(obs_f[i])
          target_ff.append(target_f[i])
    obs_p = torch.cat(obs_p, dim=0).permute(1,0,2)
    obs_s = torch.cat(obs_s, dim=0).permute(1,0,2)
    obs_m = torch.cat(obs_m, dim=0).permute(1,0,2)
    v_obs_m = torch.cat(v_obs_m, dim=0).permute(1,0,2)
    target_p = torch.cat(target_p, dim=0).permute(1,0,2)
    target_s = torch.cat(target_s, dim=0).permute(1,0,2)
    target_m = torch.cat(target_m, dim=0).permute(1,0,2)
    v_target_m = torch.cat(v_target_m, dim=0).permute(1,0,2)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [obs_p, obs_s, obs_ff, obs_m, target_p, target_s, target_ff, target_m, seq_start_end, v_obs_m, v_target_m]
    #out = [obs_p, obs_s, obs_f, obs_m, target_p, target_s, target_f, target_m, seq_start_end]
    return tuple(out)

    
def data_loader(args):
    dataset = myJAAD(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, collate_fn=my_collate, drop_last=True)

    return dataloader

