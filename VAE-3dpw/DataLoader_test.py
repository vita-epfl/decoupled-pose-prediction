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
    def __init__(self):
        print('Loading', 'test', 'data ...')
        full_path = "/home/behnam/3dpw_codes/pose-prediction/3dpw/somof_data_3dpw/"

        #3dpw_train_in.json 3dpw_train_frames_in.json 3dpw_train_out.json
        with open(os.path.join(full_path,"3dpw_test_frames_in.json"), 'r') as f:
            self.frames_in = json.load(f)

        with open(os.path.join(full_path,"3dpw_test_in.json"), 'r') as f:
            self.data_in = json.load(f)

        #print("len of data: ", len(self.data_in), len(self.data_in[0]))

        #downtown_arguing_00/image_00020.jpg
        frames_out = []
        for i in range(len(self.frames_in)):
           frames_out.append([])
           path = "/".join(self.frames_in[i][-1].split("/")[:-1])
           last = int(self.frames_in[i][-1].split("/")[-1].split("_")[-1].split(".")[0]) + 2
           for j in range(14):
              frames_out[i].append(path+"/image_{:05d}.jpg".format(last+2*j))
        self.frames_out = frames_out

        #with open(os.path.join(full_path,"3dpw_"+args.dtype+"_out.json"), 'r') as f:
        #    self.data_out = json.load(f)

    def __len__(self):
        return len(self.data_in) 
    
    
    def __getitem__(self, idx):

        obs_keypoints = torch.tensor(self.data_in[idx])   #n_ped,16,28
        #future_keypoints = torch.tensor(self.data_out[idx]) #n_ped, 14, 28

        obs_speed_keypoints = (obs_keypoints[:,1:,:] - obs_keypoints[:,:-1,:]) #n_ped, 15, 28
        #true_speed = torch.cat(((future_keypoints[:,0,:]-obs_keypoints[:,-1,:]).unsqueeze(1), future_keypoints[:,1:,:]-future_keypoints[:,:-1,:]),dim=1)

        return obs_keypoints, obs_speed_keypoints, self.frames_in[idx], self.frames_out[idx]

def my_collate(batch):

    (obs_p, obs_s, obs_f, target_f) = zip(*batch)
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
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [obs_p, obs_s, obs_ff, target_ff, seq_start_end]
    #out = [obs_p, obs_s, obs_f, obs_m, target_p, target_s, target_f, target_m, seq_start_end]
    return tuple(out)

    
def data_loader_test():
    dataset = myJAAD()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False, collate_fn=my_collate)

    return dataloader

