import torch 
import numpy as np
import time

def myVIM(pred, true, mask_):

    seq_len, batch, l = true.shape
    mask = torch.repeat_interleave(mask_, 2, dim=-1)
    displacement = torch.sum(torch.sqrt(torch.sum( (pred-true)**2*mask, dim=-1)/torch.sum(mask, dim=-1) ), dim=0)/seq_len
    NANS = torch.isnan(displacement)
    displacement[NANS] = 0
    vim = torch.sum(displacement)
    return vim


def ADE_c(pred, true):
    seq_len, batch, l = true.shape
    displacement=torch.sum(torch.sqrt(torch.sum((pred-true)**2, dim=-1) / 14. ), dim=0)/seq_len
    ade = torch.sum(displacement)
    return ade


def FDE_c(pred, true):

    seq_len, batch, l = true.shape
    displacement=torch.sqrt(torch.sum((pred[-1]-true[-1])**2, dim=-1) / 14. )
    fde = torch.sum(displacement)
    return fde

def speed2pos(preds, obs_p, dev=None):

    #dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    seq_len, batch, l = preds.shape
    pred_pos = torch.zeros(seq_len, batch, l).to(device=dev)
    current = obs_p[-1,:,:].clone().detach()
    
    for i in range(seq_len):
        pred_pos[i,:,:] = current + preds[i,:,:]
        current = pred_pos[i,:,:].clone().detach()
    return pred_pos

