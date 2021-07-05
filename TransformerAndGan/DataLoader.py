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
import utils
import json
from scipy import interpolate
import random
random.seed(10)
torch.manual_seed(10)
np.random.seed(0)

class myJAAD(torch.utils.data.Dataset):
    def __init__(self, args):
        print('Loading', args.dtype, 'data ...')

        self.args = args
        #image_size: 1280, 720
        with open(os.path.join("../somof_data_posetrack","posetrack_"+args.dtype+"_frames_in.json"), 'r') as f:
            frames = json.load(f)
        #(306, 16)
        with open(os.path.join("../somof_data_posetrack","posetrack_"+args.dtype+"_masks_in.json"), 'r') as f:
            masks_in = json.load(f)
        masks_in = np.vstack(masks_in)
        #masks_in = np.repeat(masks_in, 2, axis=-1)
        self.masks_in = torch.tensor(masks_in, dtype=torch.float32)

        with open(os.path.join("../somof_data_posetrack","posetrack_"+args.dtype+"_in.json"), 'r') as f:
            data_in = json.load(f)

        frames_in = []
        for i in range(len(data_in)):
           for j in range(len(data_in[i])):
               frames_in.append(frames[i]) 

        frames_out = []
        for i in range(len(frames_in)):
           lst = frames_in[i][-1].split("/")
           image = lst[-1]
           n = int(image.split(".")[0])
           xx = []
           for ii in range(args.output):
               path = "/".join(lst[:-1]) + "/{:06d}.jpg".format(n+ii+1)
               xx.append(path) 
           frames_out.append(xx) 

        self.frames_in = frames_in #torch.tensor(frames_in)
        self.frames_out = frames_out #torch.tensor(frames_out)

        with open(os.path.join("../somof_data_posetrack","posetrack_"+args.dtype+"_masks_out.json"), 'r') as f:
            masks_out = json.load(f)
        masks_out = np.vstack(masks_out)
        #masks_out = np.repeat(masks_out, 2, axis=-1)
        self.masks_out = torch.tensor(masks_out, dtype=torch.float32)


        with open(os.path.join("../somof_data_posetrack","posetrack_"+args.dtype+"_out.json"), 'r') as f:
            data_out = json.load(f)

        data_in = torch.tensor(np.vstack(data_in), dtype=torch.float32)
        data_out = torch.tensor(np.vstack(data_out), dtype=torch.float32)



        self.data_in = data_in#*self.masks_in
        self.data_out = data_out#*self.masks_out

        if(args.scale):
            T = self.data_in.shape[1]
            self.data_in = self.data_in.reshape(-1, T, 14, 2) 
            self.data_in[:,:,:,0] /= 1280.
            self.data_in[:,:,:,1] /= 720. 
            self.data_in = self.data_in.reshape(-1, T, 28) 

            T = self.data_out.shape[1]
            self.data_out = self.data_out.reshape(-1, T, 14, 2) 
            self.data_out[:,:,:,0] /= 1280.
            self.data_out[:,:,:,1] /= 720.
            self.data_out = self.data_out.reshape(-1, T, 28) 
       
        

    def __len__(self):
        assert(self.data_in.shape[0] == self.data_out.shape[0])
        return self.data_in.shape[0]
    
    
    def __getitem__(self, idx):

     
        obs_keypoints = self.data_in[idx]
        future_keypoints = self.data_out[idx]

        obs_speed_keypoints = (obs_keypoints[1:] - obs_keypoints[:-1])
        true_speed = torch.cat(((future_keypoints[0]-obs_keypoints[-1]).unsqueeze(0), future_keypoints[1:]-future_keypoints[:-1]))
        return obs_keypoints, obs_speed_keypoints, self.frames_in[idx], self.masks_in[idx], \
               future_keypoints, true_speed, self.frames_out[idx], self.masks_out[idx]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    
def data_loader(args):
    dataset = myJAAD(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    return dataloader
