import torch
torch.manual_seed(0) 
import numpy as np
from PIL import Image, ImageDraw
import os

JOINTS = [
    (0, 1),  # head_top -> head_center
    (1, 2),  # head_center -> neck
    (2, 3),  # neck -> right_clavicle
    (3, 4),  # right_clavicle -> right_shoulder
    (4, 5),  # right_shoulder -> right_elbow
    (5, 6),  # right_elbow -> right_wrist
    (2, 7),  # neck -> left_clavicle
    (7, 8),  # left_clavicle -> left_shoulder
    (8, 9),  # left_shoulder -> left_elbow
    (9, 10),  # left_elbow -> left_wrist
    (2, 11),  # neck -> spine0
    (11, 12),  # spine0 -> spine1
    (12, 13),  # spine1 -> spine2
    (13, 14),  # spine2 -> spine3
    (14, 15),  # spine3 -> spine4
    (15, 16),  # spine4 -> right_hip
    (16, 17),  # right_hip -> right_knee
    (17, 18),  # right_knee -> right_ankle
    (15, 19),  # spine4 -> left_hip
    (19, 20),  # left_hip -> left_knee
    (20, 21)  # left_knee -> left_ankle
]
PATH = "/scratch/izar/saeedsa/pose-prediction/JTA_viz1"
WIDTH = 2

def visu(obs_p, obs_m, obs_f, pred, true, masks, scenes, seq_start_end, prefix, epoch, idx):

    obs_len, batch, l = obs_p.shape
    obs_p = obs_p.view(obs_len, batch, l//2, 2) 

    pred_len, batch, l = pred.shape
    pred = pred.view(pred_len, batch, l//2, 2) 
    true = true.view(pred_len, batch, l//2, 2) 

    random_images = torch.randint(0, len(seq_start_end)-1, (10,))
    print("random numbers: ", random_images)
   

    #loop over pedestrians
    for _, (start, end) in enumerate(seq_start_end[random_images]):
        for time in range(pred_len):
            im =Image.open(os.path.join(prefix, scenes[start][time].replace(".json", ".jpg")))
            img = ImageDraw.Draw(im)
            for ped in range(start, end):
                if( 1 in masks[time, ped] ): continue 
                #if( (masks[time, ped] == 1).all() ): continue 
                for joint in JOINTS:
                    if(joint[0] == 1 or joint[1] == 1): continue
                    pred_points = (pred[time, ped, joint[0],0], pred[time, ped, joint[0], 1], pred[time, ped, joint[1],0], pred[time, ped, joint[1], 1])
                    true_points = (true[time, ped, joint[0],0], true[time, ped, joint[0], 1], true[time, ped, joint[1],0], true[time, ped, joint[1], 1])
                    img.line(pred_points, fill="red", width=WIDTH)
                    img.line(true_points, fill="blue", width=WIDTH)
            #im.save(PATH+"/{:04d}-{:04d}-{:04d}-{:04d}-{}.png".format(epoch, idx, ped, time, scenes[start][time].replace("/","_")))
            im.save(PATH+"/{:04d}-{:04d}--{}.png".format(epoch, time, scenes[start][time].replace("/","_")))

    obs_p = obs_p.view(obs_len, batch, l) 
    pred = pred.view(pred_len, batch, l) 
    true = true.view(pred_len, batch, l) 

def visualization1(pred, true, masks, scenes, prefix, epoch, idx):

    pred_len, batch, l = pred.shape
    pred = pred.view(pred_len, batch, l//2, 2) 
    true = true.view(pred_len, batch, l//2, 2) 

    random_images = torch.randint(0, batch-1, (10,))

    for _ in random_images:
        if(1 in masks[:,_]): 
            #print("deb: ", 0 in masks[:,_], masks[:, _].shape)
            continue
        for time in range(pred_len):
            im =Image.open(prefix + scenes[_][time].replace(".json", ".jpg"))
            img = ImageDraw.Draw(im)
            for joint in JOINTS:
                pred_points = (pred[time, _, joint[0],0], pred[time, _, joint[0], 1], pred[time, _, joint[1],0], pred[time, _, joint[1], 1])
                true_points = (true[time, _, joint[0],0], true[time, _, joint[0], 1], true[time, _, joint[1],0], true[time, _, joint[1], 1])
                img.line(pred_points, fill="red", width=WIDTH)
                img.line(true_points, fill="blue", width=WIDTH)
            im.save(PATH+"/{:04d}-{:04d}-{:04d}-{:04d}-{}.png".format(epoch, idx,_,time, scenes[_][time].replace("/","_")))

    pred = pred.view(pred_len, batch, l) 
    true = true.view(pred_len, batch, l) 

def myVIM(pred, true):

    seq_len, batch, l = true.shape
    displacement = torch.sum(torch.sqrt(torch.sum( (pred-true)**2, dim=-1)/22 ), dim=0)/seq_len
    vim = torch.sum(displacement)
    return vim


def ADE_c(pred, true):

    seq_len, batch, l = true.shape
    displacement=torch.sum(torch.sqrt(torch.sum((pred-true)**2, dim=-1) / 22. ), dim=0)/seq_len
    ade = torch.sum(displacement)
    return ade


def FDE_c(pred, true):

    seq_len, batch, l = true.shape
    displacement=torch.sqrt(torch.sum((pred[-1]-true[-1])**2, dim=-1) / 22.)
    fde = torch.sum(displacement)
    return fde

def speed2pos(preds, obs_p):
     
    seq_len, batch, l = preds.shape
    pred_pos = torch.zeros(seq_len, batch, l).to('cuda' if torch.cuda.is_available() else 'cpu')
    current = obs_p[-1,:,:].clone().detach()
    for i in range(seq_len):
        pred_pos[i,:,:] = current + preds[i,:,:]
        current = pred_pos[i,:,:].clone().detach()

    return pred_pos

def visualize(path, seq_in, seq_out, seq_pred):

    viewer = viz.Seq3DPose(window=1.0)
    viewer.center(seq_in[..., 4:6, 0].mean(), seq_in[..., 4:6, 1].mean(), seq_in[..., 4:6, 2].mean())
    viewer.view(seq_in, 'Input', prefix=path)
    viewer.view(seq_pred, 'Prediction', prefix=path)
    viewer.view(seq_out, 'GroundTruth', prefix=path)
