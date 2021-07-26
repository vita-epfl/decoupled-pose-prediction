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

from PIL import Image, ImageDraw
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import args_module
import random

random.seed(10)
torch.manual_seed(10)
np.random.seed(0)

args = args_module.args()
scale = args.scale #True
scalex = args.scalex #1280
scaley = args.scaley #720

def plot_joints(img, keypoints, masks, scale, scalex=1280, scaley=720, color='red', width=10):

    keypoints = keypoints.reshape(14, 2)
    if(scale):
        keypoints[:,0] *= scalex #1280.
        keypoints[:,1] *= scaley #720.

   # joints = [(0,1),(1,2),(1,3),(2,4),(3,5),(4,6),(5,7),(8,9), \
   #            (1,8),(1,9),(10,11),(8,10), (9, 11), (10, 12), (11,13)]
    connections = {0:[1], 1:[2,3,8,9], 2:[4], 4:[6], 3:[5], 5: [7], 8: [9,10], 10: [12], 9:[11], 11:[13]}
    lines = []
    for i in connections.keys():
        for j in connections[i]:
            if(masks[i] and masks[j]):
               line = [(keypoints[i][0], keypoints[i][1]), (keypoints[j][0], keypoints[j][1])]
               lines.append(line)

    for line in lines:
        img.line(line, fill =color, width = width)
          
    #for i, (x, y) in enumerate(keypoints.reshape(14, 2)):
    #   if(masks[i]):
    #       if(i ==0 ):
    #           img.ellipse([(x,y), (x+10,y+10)], fill='blue', outline='blue', width=width)
    #       else:
    #           img.ellipse([(x,y), (x+10,y+10)], fill=color, outline=color, width=width)

    if(scale):
        keypoints[:,0] /= scalex #1280.
        keypoints[:,1] /= scaley #720.
    keypoints = keypoints.reshape(28)


def plotter(obs_frames, obs_p, obs_masks, target_frames, preds_p, preds_masks, target_p, target_masks, e, idx, rnd, width=10):
      obs_scenes = [obs_frames[rnd][i] for i in range(args.input)]
      fig = plt.figure(figsize=(16,16))
      grid = ImageGrid(fig, 111,  # similar to subplot(111)
                       nrows_ncols=(2,2),  # creates 2x2 grid of axes
                       axes_pad=0.1,  # pad between axes in inch.
                       )
      #fig.suptitle('Observed scenes', fontsize=200)
      for i, ax in enumerate(grid):
          # Iterating over the grid returns the Axes.
          with open("/scratch/izar/parsaeif/posetrack/"+obs_scenes[3*i+1],'rb') as f:
              #print("deb: ", obs_scenes[3*i+1], obs_kp[0][3*i+1].shape)
              image=Image.open(f)
              img = ImageDraw.Draw(image)
              plot_joints(img, obs_p[3*i+1][rnd], obs_masks[3*i+1][rnd], scale, scalex, scaley, color='green', width=width)
              #plot_keypoints(img, obs_pose[3*i+1], color='green')
              ax.imshow(image)
          #ax[i%2][i//2].axis('off')
      
      plt.savefig('/scratch/izar/parsaeif/lstm_pooling/obs_pose_e{:04d}-idx{:04d}-rnd{:04d}.png'.format(e, idx, rnd))

      target_scenes = [target_frames[rnd][j] for j in range(14)]
      fig = plt.figure(figsize=(16,16))
      grid = ImageGrid(fig, 111,  # similar to subplot(111)
                       nrows_ncols=(2,2),  # creates 2x2 grid of axes
                       axes_pad=0.1,  # pad between axes in inch.
                       )
      #fig.suptitle('Observed scenes', fontsize=200)
      for i, ax in enumerate(grid):
          # Iterating over the grid returns the Axes.
          #print(f"deb: {i} {target_scenes[3*i+1]}")
          #with open("/scratch/izar/parsaeif/posetrack/"+target_scenes[3*i+1],'rb') as f:
          with open("/scratch/izar/parsaeif/posetrack/"+target_scenes[3*i+1],'rb') as f:
              #print("deb: ", obs_scenes[3*i+1], obs_kp[0][3*i+1].shape)
              image=Image.open(f)
              img = ImageDraw.Draw(image)
              plot_joints(img, preds_p[3*i+1][rnd], preds_masks[3*i+1][rnd], scale, scalex, scaley, color='red', width=width)
              plot_joints(img, target_p[3*i+1][rnd], target_masks[3*i+1][rnd], scale, scalex, scaley, color='green', width=width)
              #plot_keypoints(img, obs_pose[3*i+1], color='green')
              ax.imshow(image)
          #ax[i%2][i//2].axis('off')

      plt.savefig('/scratch/izar/parsaeif/lstm_pooling/predicted_pose_e{:04d}-idx{:04d}-rnd{:04d}.png'.format(e,idx, rnd))



def myVIM(pred, true):

    #pred => seq_len, batch, 28
    #true => seq_len, batch, 28
    #mask => seq_len, batch, 14
    seq_len, batch, l = true.shape

    displacement = torch.sum(torch.sqrt(torch.sum( (pred-true)**2, dim=-1)/13 ), dim=0)/seq_len
    #NANS = torch.isnan(displacement)
    #displacement[NANS] = 0
    vim = torch.sum(displacement)

    return vim


def ADE_c(pred, true):

    seq_len, batch, l = true.shape

    displacement=torch.sum(torch.sqrt(torch.sum((pred-true)**2, dim=-1) / 13. ), dim=0)/seq_len
    ade = torch.sum(displacement)

    return ade


def FDE_c(pred, true):

    seq_len, batch, l = true.shape

    displacement=torch.sqrt(torch.sum((pred[-1]-true[-1])**2, dim=-1) / 13.)
    fde = torch.sum(displacement)

    return fde

def speed2pos(preds, obs_p):
     
    seq_len, batch, l = preds.shape
    pred_pos = torch.zeros(seq_len, batch, l).to('cpu')
    current = obs_p[-1,:,:]
    
    for i in range(seq_len):
        pred_pos[i,:,:] = current + preds[i,:,:]
        current = pred_pos[i,:,:]
        
    #pred_pos = torch.min(pred_pos, 1000*torch.ones(seq_len, batch, l, device='cuda'))
    #pred_pos = torch.max(pred_pos,-1000*torch.ones(seq_len, batch, l, device='cuda'))
        
    return pred_pos

def v2pos(preds, obs_p):
     
    #preds[0] = 0.5*(pred_pos[1]-obs_p[-1]) => pred_pos[1]=2preds[0] + obs_p[-1]
    #preds[1] = 0.5*(pred_pos[2]-pred_pos[0]) => pred_pos[2]=2preds[1]+pred_pos[0]
    #preds[2] = 0.5*(pred_pos[3]-pred_pos[1])
    #          ...
    #preds[12] = 0.5*(pred_pos[13]-pred_pos[11])


    seq_len, batch, l = preds.shape
    pred_pos = torch.zeros(seq_len, batch, l).to('cpu')
    current = obs_p[-1,:,:]
    
    for i in range(seq_len):
        pred_pos[i,:,:] = current + preds[i,:,:]
        current = pred_pos[i,:,:]
        
    #pred_pos = torch.min(pred_pos, 1000*torch.ones(seq_len, batch, l, device='cuda'))
    #pred_pos = torch.max(pred_pos,-1000*torch.ones(seq_len, batch, l, device='cuda'))
        
    return pred_pos

