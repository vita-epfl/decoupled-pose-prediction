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
import pickle
STRIDE = 30

class myJAAD(torch.utils.data.Dataset):
    def __init__(self, args):
        print('Loading', args.dtype, 'data ...')
        full_path = "/home/behnam/3dpw_codes/pose-prediction/3dpw/somof_data_3dpw/"

        #with open("extrinsic.pkl", 'rb') as f:
        #    self.info = pickle.load(f)
        #self.info["NotExist"] = np.eye(4, dtype=np.float32)


        #print("DEB: ", self.info.keys(), self.info["NotExist"])
        #print(type(self.info["NotExist"]))
        #print(type(self.info["outdoors_climbing_00/image_02450.jpg"]))

        self.args = args
        #3dpw_train_in.json 3dpw_train_frames_in.json 3dpw_train_out.json
        with open(os.path.join(full_path,"3dpw_"+args.dtype+"_frames_in_stride_"+str(STRIDE)+".json"), 'r') as f:
            self.frames_in = json.load(f)

        with open(os.path.join(full_path,"3dpw_"+args.dtype+"_in_stride_"+str(STRIDE)+".json"), 'r') as f:
            self.data_in = json.load(f)

        #downtown_arguing_00/image_00020.jpg
        frames_out = []
        for i in range(len(self.frames_in)):
           frames_out.append([])
           path = "/".join(self.frames_in[i][-1].split("/")[:-1])
           last = int(self.frames_in[i][-1].split("/")[-1].split("_")[-1].split(".")[0]) + 2
           for j in range(14):
              frames_out[i].append(path+"/image_{:05d}.jpg".format(last+2*j))
        self.frames_out = frames_out

        with open(os.path.join(full_path,"3dpw_"+args.dtype+"_out_stride_"+str(STRIDE)+".json"), 'r') as f:
            self.data_out = json.load(f)

    def __len__(self):
        return len(self.data_in) 
    
        
    def prepocess(self, obs_p, f_in):
        rotation = torch.tensor([], dtype=torch.float32, device='cpu')
        translation = torch.tensor([], dtype=torch.float32,  device='cpu')
        n_ped, seq_len, l = obs_p.shape
        for t in range(seq_len):
            frame = f_in[t]
            try:
                mat = self.info[frame]
            except:
                mat = self.info["NotExist"]
                #try:
                #    frame = f_in[t-1]
                #except:
                    #try:
                    #    frame = f_in[t+1]
                    #except:
                    #    frame = "NotExist"

            rot = torch.tensor(mat[:3,:3])
            tr = torch.tensor(mat[:3,3])
            rotation = torch.cat((rotation, rot.unsqueeze(0)), dim=0)
            translation = torch.cat((translation, tr.unsqueeze(0)), dim=0)
        #translation => (16,3) => (1, 16, 1, 3) => (2, 16, 13, 3)
        #rotation => (16,3,3) => (1, 16, 1, 3, 3) => (2, 16, 13, 3, 3)
        rotation = rotation.unsqueeze(0).unsqueeze(2).repeat(n_ped, 1, 13, 1, 1)
        translation = translation.unsqueeze(0).unsqueeze(2).repeat(n_ped, 1, 13, 1)
        #print("deb: ", rotation.shape, translation.shape) #16, 3, 3    16, 3
        obs_p = obs_p.view(n_ped, seq_len, l//3, 3)
        obs_p = obs_p 
        obs_p = obs_p.unsqueeze(4)
        obs_p = torch.matmul(rotation.double(), obs_p.double()) #(2, 16, 13, 3, 3), (2, 16, 13, 3, 1) => (2, 16, 13, 3, 1)
        obs_p = obs_p.squeeze(4) + translation
        obs_p = obs_p.view(n_ped, seq_len, l)
        return obs_p

    
    def __getitem__(self, idx):


        obs_p = torch.tensor(self.data_in[idx])   #n_ped,16,28
        target_p = torch.tensor(self.data_out[idx]) #n_ped, 14, 28

        #f_in = self.frames_in[idx] #16
        #f_out = self.frames_out[idx] #14
        #obs_p = self.prepocess(obs_p, f_in)
        #target_p = self.prepocess(target_p, f_out)

        obs_s = (obs_p[:,1:,:] - obs_p[:,:-1,:]) #n_ped, 15, 28
        target_s = torch.cat(((target_p[:,0,:]-obs_p[:,-1,:]).unsqueeze(1), target_p[:,1:,:]-target_p[:,:-1,:]),dim=1)

        return obs_p, obs_s, self.frames_in[idx], target_p, target_s, self.frames_out[idx]

def my_collate(batch):

    #(ref, obs_p, obs_s, obs_f, target_p, target_s, target_f) = zip(*batch)
    (obs_p, obs_s, obs_f, target_p, target_s, target_f) = zip(*batch)
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
    #ref = torch.cat(ref, dim=0).permute(1,0,2)
    obs_s = torch.cat(obs_s, dim=0).permute(1,0,2)
    target_p = torch.cat(target_p, dim=0).permute(1,0,2)
    target_s = torch.cat(target_s, dim=0).permute(1,0,2)
    seq_start_end = torch.LongTensor(seq_start_end)
    #out = [obs_p, obs_s, obs_ff, target_p, target_s, target_ff, seq_start_end]
    out = [obs_p, obs_s, obs_f, target_p, target_s, target_f, seq_start_end]
    return tuple(out)

    
def data_loader(args):
    dataset = myJAAD(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, collate_fn=my_collate, drop_last=False)

    return dataloader

def data_loader_val(args):
    dataset = myJAAD(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=dataset.__len__(), shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, collate_fn=my_collate, drop_last=False)

    return dataloader

