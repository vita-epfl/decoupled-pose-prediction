import torch 
import numpy as np
import viz

def myVIM(pred, true):

    seq_len, batch, l = true.shape
    displacement = torch.sum(torch.sqrt(torch.sum( (pred-true)**2, dim=-1)/13 ), dim=0)/seq_len
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

def speed2pos(preds, obs_p, dev=None):
    #dev = 'cuda' if torch.cuda.is_available() else 'cpu'
     
    seq_len, batch, l = preds.shape
    pred_pos = torch.zeros(seq_len, batch, l).to(dev)
    current = obs_p[-1,:,:].clone().detach()
    for i in range(seq_len):
        pred_pos[i,:,:] = current + preds[i,:,:]
        current = pred_pos[i,:,:].clone().detach()

    return pred_pos

def visualize(path, seq_in, seq_out, seq_pred):

    viewer = viz.Seq3DPose(window=1.0)
    viewer.center(seq_in.cpu()[..., 4:6, 0].mean(), seq_in.cpu()[..., 4:6, 1].mean(), seq_in.cpu()[..., 4:6, 2].mean())
    viewer.view(seq_in.cpu(), 'Input', prefix=path)
    viewer.view(seq_pred.cpu(), 'Prediction', prefix=path)
    viewer.view(seq_out.cpu(), 'GroundTruth', prefix=path)

