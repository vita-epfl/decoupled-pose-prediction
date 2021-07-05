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
import numpy as np
import random
random.seed(10)
torch.manual_seed(10)
np.random.seed(0)

class myJAAD(torch.utils.data.Dataset):
    def __init__(self, args):
        print('Loading', args.dtype, 'data ...')

        self.args = args
        with open(os.path.join("somof_data_posetrack","posetrack_"+args.dtype+"_frames_in.json"), 'r') as f:
            frames = json.load(f)
        with open(os.path.join("somof_data_posetrack","posetrack_"+args.dtype+"_masks_in.json"), 'r') as f:
            masks_in = json.load(f)

        masks_in = np.vstack(masks_in)
        #masks_in = np.repeat(masks_in, 2, axis=-1)
        self.masks_in = torch.tensor(masks_in, dtype=torch.float32)

        with open(os.path.join("somof_data_posetrack","posetrack_"+args.dtype+"_in.json"), 'r') as f:
            data_in = json.load(f)

        frames_in = []
        info = []
        for i in range(len(data_in)):
           info.append(len(data_in[i]))
           for j in range(len(data_in[i])):
               frames_in.append(frames[i]) 
        self.info = info

        frames_out = []
        for i in range(len(frames_in)):
           lst = frames_in[i][-1].split("/")
           image = lst[-1]
           n = int(image.split(".")[0])
           for ii in range(args.output):
               path = "/".join(lst[:-1]) + "/{:06d}.jpg".format(n+ii+1)
               frames_out.append(path) 

        self.frames_in = frames_in 
        self.frames_out = frames_out 

        data_in = torch.tensor(np.vstack(data_in), dtype=torch.float32)
        self.data_in = data_in #*self.masks_in

        T = self.data_in.shape[1]
        self.data_in = self.data_in.reshape(-1, T, 14, 2) 
        self.data_in[:,:,:,0] /= 1280.
        self.data_in[:,:,:,1] /= 720. 
        self.data_in = self.data_in.reshape(-1, T, 28) 
        

    def __len__(self):
        #assert(self.data_in.shape[0] == self.data_out.shape[0])
        return self.data_in.shape[0]
    
    
    def __getitem__(self, idx):

     
        obs_keypoints = self.data_in[idx]
        obs_speed_keypoints = (obs_keypoints[1:] - obs_keypoints[:-1])
        return obs_keypoints, obs_speed_keypoints, self.frames_in[idx], self.masks_in[idx], self.frames_out[idx]


    
def data_loader(args):
    dataset = myJAAD(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=dataset.__len__(), shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers)

    return dataloader, dataset.info
