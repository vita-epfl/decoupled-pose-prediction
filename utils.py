import numpy as np
import torch

from PIL import Image, ImageDraw
import args_module
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


args = args_module.args()
device = args.device
scale  = args.scale
scalex = args.scalex
scaley = args.scaley


def myVIM(pred, true, mask):
    n_batch, seq_len, l = true.shape

    if(scale):
        true = true.view(n_batch, seq_len, 14, 2)
        pred = pred.view(n_batch, seq_len, 14, 2)
        true[:,:,:,0] *= scalex #1280 
        true[:,:,:,1] *= scaley #720 
        pred[:,:,:,0] *= scalex #1280 
        pred[:,:,:,1] *= scaley #720 
        true = true.view(n_batch, seq_len, 28)
        pred = pred.view(n_batch, seq_len, 28)


    #displacement => (n_batch, seq_len ) 
    #mask = np.repeat(mask, 2, axis=-1)
    mask = torch.repeat_interleave(mask, 2, dim=-1)
    displacement = torch.sqrt(torch.sum( (pred-true)**2*mask,dim=-1)/torch.sum(mask, dim=-1) )
    NANS = torch.isnan(displacement)
    displacement[NANS] = 0
    #vim = torch.sum(displacement)
    vim = torch.mean(displacement)

    

    if(scale):
        true = true.view(n_batch, seq_len, 14, 2)
        pred = pred.view(n_batch, seq_len, 14, 2)
        true[:,:,:,0] /= scalex #1280 
        true[:,:,:,1] /= scaley #720 
        pred[:,:,:,0] /= scalex #1280 
        pred[:,:,:,1] /= scaley #720 
        true = true.view(n_batch, seq_len, 28)
        pred = pred.view(n_batch, seq_len, 28)

    return vim
    

def VIM(GT, pred, dataset_name, mask):
    """
    Visibilty Ignored Metric
    Inputs:
        GT: Ground truth data - array of shape (pred_len, #joint*(2D/3D))
        pred: Prediction data - array of shape (pred_len, #joint*(2D/3D))
        dataset_name: Dataset name
        mask: Visibility mask of pos - array of shape (pred_len, #joint)
    Output:
        errorPose:
    """

    gt_i_global = np.copy(GT.detach().cpu().numpy())

    if dataset_name == "posetrack":
        #mask = np.repeat(mask, 2, axis=-1)
        errorPose = np.power(gt_i_global - pred.detach().cpu().numpy(), 2) * mask.detach().cpu().numpy()
        #get sum on joints and remove the effect of missing joints by averaging on visible joints
        errorPose = np.sqrt(np.divide(np.sum(errorPose, 1), np.sum(mask.detach().cpu().numpy(),axis=1)))
        where_are_NaNs = np.isnan(errorPose)
        errorPose[where_are_NaNs] = 0
    else:   #3dpw
        errorPose = np.power(gt_i_global - pred.detach().cpu().numpy(), 2)
        errorPose = np.sum(errorPose, 1)
        errorPose = np.sqrt(errorPose)
    return np.mean(errorPose)




def ADE_c(pred, true):
    
    #print(f'pred {pred.device} true {true.device} ' )
    displacement = torch.sqrt((pred[:,:,0]-true[:,:,0])**2 + (pred[:,:,1]-true[:,:,1])**2)
    
    ade = torch.mean(displacement)
    
    return ade


def FDE_c(pred, true):
    displacement = torch.sqrt((pred[:,-1,0]-true[:,-1,0])**2 + (pred[:,-1,1]-true[:,-1,1])**2)
    fde = torch.mean(displacement)
    
    return fde

def ADE_keypoints(pred, true):
    
    #print(f'pred {pred.device} true {true.device} ' )
    n_batch, seq_len, l = pred.shape
    if(scale):
        true = true.view(n_batch, seq_len, l//2, 2)
        pred = pred.view(n_batch, seq_len, l//2, 2)
        true[:,:,:,0] *= scalex #1280 
        true[:,:,:,1] *= scaley #720 
        pred[:,:,:,0] *= scalex #1280 
        pred[:,:,:,1] *= scaley #720 
        true = true.view(n_batch, seq_len, l)
        pred = pred.view(n_batch, seq_len, l)
         

    displacement = torch.sqrt(torch.sum( (pred[:,:,:]-true[:,:,:])**2,dim=-1)/ (l//2) )
    
    ade = torch.mean(displacement)
    
    if(scale):
        true = true.view(n_batch, seq_len, l//2, 2)
        pred = pred.view(n_batch, seq_len, l//2, 2)
        true[:,:,:,0] /= scalex #1280 
        true[:,:,:,1] /= scaley #720 
        pred[:,:,:,0] /= scalex #1280 
        pred[:,:,:,1] /= scaley #720 
        true = true.view(n_batch, seq_len, l)
        pred = pred.view(n_batch, seq_len, l)

    return ade

def FDE_keypoints(pred, true):
    n_batch, seq_len, l = true.shape

    if(scale):
        true = true.view(n_batch, seq_len, l//2, 2)
        pred = pred.view(n_batch, seq_len, l//2, 2)
        true[:,:,:,0] *= scalex #1280 
        true[:,:,:,1] *= scaley #720 
        pred[:,:,:,0] *= scalex #1280 
        pred[:,:,:,1] *= scaley #720 
        true = true.view(n_batch, seq_len, l)
        pred = pred.view(n_batch, seq_len, l)

    displacement = torch.sqrt(torch.sum( (pred[:,-1,:]-true[:,-1,:])**2, dim=-1) / (l//2))
    fde = torch.mean(displacement)

    if(scale):
        true = true.view(n_batch, seq_len, l//2, 2)
        pred = pred.view(n_batch, seq_len, l//2, 2)
        true[:,:,:,0] /= scalex #1280 
        true[:,:,:,1] /= scaley #720 
        pred[:,:,:,0] /= scalex #1280 
        pred[:,:,:,1] /= scaley #720 
        true = true.view(n_batch, seq_len, l)
        pred = pred.view(n_batch, seq_len, l)

    return fde


def AAE(pred, true):
    pred_a = pred[:,:,2] * pred[:,:,3]
    true_a = true[:,:,2] * true[:,:,3]
    
    area_error = torch.abs(pred_a - true_a)
    aae = torch.mean(area_error)
    
    return aae


def AIOU(pred, true):
    pred[:,:,0] *= 1920
    pred[:,:,2] *= 1920
    pred[:,:,1] *= 1080
    pred[:,:,3] *= 1080
    true[:,:,0] *= 1920
    true[:,:,2] *= 1920
    true[:,:,1] *= 1080
    true[:,:,3] *= 1080
    min_pred = pred[:,:,:2]-pred[:,:,2:]/2
    max_pred = pred[:,:,:2]+pred[:,:,2:]/2
    min_true = true[:,:,:2]-true[:,:,2:]/2
    max_true = true[:,:,:2]+true[:,:,2:]/2
    
    min_inter = torch.max(min_pred, min_true)
    max_inter = torch.min(max_pred, max_true)

    interArea = torch.max(torch.zeros(min_inter.shape[0],min_inter.shape[1]).to(device=device), (max_inter[:,:,0]-min_inter[:,:,0])) *\
                torch.max(torch.zeros(max_inter.shape[0],max_inter.shape[1]).to(device=device), (max_inter[:,:,1]-min_inter[:,:,1]))

    pred_a = pred[:,:,2] * pred[:,:,3]
    true_a = true[:,:,2] * true[:,:,3]
    
    iou = torch.mean(interArea / (pred_a + true_a - interArea))
    pred[:,:,0] /= 1920
    pred[:,:,2] /= 1920
    pred[:,:,1] /= 1080
    pred[:,:,3] /= 1080
    true[:,:,0] /= 1920
    true[:,:,2] /= 1920
    true[:,:,1] /= 1080
    true[:,:,3] /= 1080
    return float(iou)


def FIOU(pred, true):
    pred[:,:,0] *= 1920
    pred[:,:,2] *= 1920
    pred[:,:,1] *= 1080
    pred[:,:,3] *= 1080
    true[:,:,0] *= 1920
    true[:,:,2] *= 1920
    true[:,:,1] *= 1080
    true[:,:,3] *= 1080
    min_pred = pred[:,-1,:2]-pred[:,-1,2:]/2
    max_pred = pred[:,-1,:2]+pred[:,-1,2:]/2
    min_true = true[:,-1,:2]-true[:,-1,2:]/2
    max_true = true[:,-1,:2]+true[:,-1,2:]/2
    
    min_inter = torch.max(min_pred, min_true)
    max_inter = torch.min(max_pred, max_true)
    
    interArea = torch.max(torch.zeros(min_inter.shape[0]).to(device=device), (max_inter[:,0]-min_inter[:,0])) * \
                torch.max(torch.zeros(max_inter.shape[0]).to(device=device), (max_inter[:,1]-min_inter[:,1]))

    pred_a = pred[:,-1,2] * pred[:,-1,3]
    true_a = true[:,-1,2] * true[:,-1,3]
    
    iou = torch.mean(interArea / (pred_a + true_a - interArea))
    pred[:,:,0] /= 1920
    pred[:,:,2] /= 1920
    pred[:,:,1] /= 1080
    pred[:,:,3] /= 1080
    true[:,:,0] /= 1920
    true[:,:,2] /= 1920
    true[:,:,1] /= 1080
    true[:,:,3] /= 1080
    return float(iou)


def compute_center(row):
    row['x'] = row['x'] + row['w']/2
    row['y'] = row['y'] + row['h']/2
    
    return row


def speed2pos(preds, obs_p):
    pred_pos = torch.zeros(preds.shape[0], preds.shape[1], 4).to(device=device)
    current = obs_p[:,-1,:]
    for i in range(preds.shape[1]):
        pred_pos[:,i,:] = current + preds[:,i,:]
        current = pred_pos[:,i,:]
        
    pred_pos[:,:,0] = torch.min(pred_pos[:,:,0], torch.ones(pred_pos.shape[0], pred_pos.shape[1], device=device))
    pred_pos[:,:,1] = torch.min(pred_pos[:,:,1], torch.ones(pred_pos.shape[0], pred_pos.shape[1], device=device))
    pred_pos[:,:,0] = torch.max(pred_pos[:,:,0], torch.zeros(pred_pos.shape[0], pred_pos.shape[1], device=device))
    pred_pos[:,:,1] = torch.max(pred_pos[:,:,1], torch.zeros(pred_pos.shape[0], pred_pos.shape[1], device=device))
        
    return pred_pos

def speed2bodypose(preds, obs_p):
    pred_pos = torch.zeros(preds.shape[0], preds.shape[1], preds.shape[-1]).to(device=device)
    current = obs_p[:,-1,:]
    for i in range(preds.shape[1]):
        pred_pos[:,i,:] = current + preds[:,i,:]
        current = pred_pos[:,i,:]
        
    if(scale): 
       pred_pos[:,:,0] = torch.min(pred_pos[:,:,0], 1.*torch.ones(pred_pos.shape[0], pred_pos.shape[1], device=device))
       pred_pos[:,:,1] = torch.min(pred_pos[:,:,1], 1.*torch.ones(pred_pos.shape[0], pred_pos.shape[1], device=device))
    else:
       pred_pos[:,:,0] = torch.min(pred_pos[:,:,0], scalex*torch.ones(pred_pos.shape[0], pred_pos.shape[1], device=device))
       pred_pos[:,:,1] = torch.min(pred_pos[:,:,1], scaley*torch.ones(pred_pos.shape[0], pred_pos.shape[1], device=device))
    pred_pos[:,:,0] = torch.max(pred_pos[:,:,0], torch.zeros(pred_pos.shape[0], pred_pos.shape[1], device=device))
    pred_pos[:,:,1] = torch.max(pred_pos[:,:,1], torch.zeros(pred_pos.shape[0], pred_pos.shape[1], device=device))
        
    return pred_pos


def compute_corners(bb):
    x_low_left = int(bb[0] - bb[2]/2)
    y_low_left = int(bb[1] - bb[3]/2)
    x_high_right = int(bb[0] + bb[2]/2)
    y_high_right = int(bb[1] + bb[3]/2)
    
    return (x_low_left, y_low_left), (y_high_right, y_high_right)


def drawrect(drawcontext, bb, width=5):
    (x1, y1), (x2, y2) = bb
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill="red", width=width)


#def plot_joints(img, keypoints, masks, color='red', width=5.):
#
#    keypoints = keypoints.reshape(14, 2)
#    if(scale):
#        keypoints[:,0] *= scalex #1280.
#        keypoints[:,1] *= scaley #720.
#
#
#    for i, (x, y) in enumerate(keypoints.reshape(14, 2)):
#       if(masks[i]):
#           if(i ==0 ):
#               img.ellipse([(x,y), (x+10,y+10)], fill='blue', outline='blue', width=width)
#           else:
#               img.ellipse([(x,y), (x+10,y+10)], fill=color, outline=color, width=width)
#
#    if(scale):
#       keypoints[:,0] /= scalex #1280.
#       keypoints[:,1] /= scaley #720.
#    keypoints = keypoints.reshape(28)

def plot_joints(img, keypoints, masks, scale, scalex=1280, scaley=720, color='red', width=10):

    keypoints = keypoints.reshape(14, 2)
    if(scale):
        keypoints[:,0] *= scalex 
        keypoints[:,1] *= scaley 

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
          
    if(scale):
        keypoints[:,0] /= scalex 
        keypoints[:,1] /= scaley 
    keypoints = keypoints.reshape(28)


def plotter(obs_frames, obs_p, obs_masks, target_frames, preds_p, preds_masks, target_p, target_masks, e, idx, rnd, width=10):
      obs_scenes = [obs_frames[i][rnd] for i in range(args.input)]
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
              plot_joints(img, obs_p[rnd][3*i+1], obs_masks[rnd][3*i+1], scale, scalex, scaley, color='green', width=width)
              #plot_keypoints(img, obs_pose[3*i+1], color='green')
              ax.imshow(image)
          #ax[i%2][i//2].axis('off')
      
      plt.savefig('/scratch/izar/parsaeif/lstm_res_visul/obs_pose_e{:04d}-idx{:04d}-rnd{:04d}.png'.format(e, idx, rnd))

      target_scenes = [target_frames[j][rnd] for j in range(14)]
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
              plot_joints(img, preds_p[rnd][3*i+1], preds_masks[rnd][3*i+1], scale, scalex, scaley, color='red', width=width)
              plot_joints(img, target_p[rnd][3*i+1], target_masks[rnd][3*i+1], scale, scalex, scaley, color='green', width=width)
              #plot_keypoints(img, obs_pose[3*i+1], color='green')
              ax.imshow(image)
          #ax[i%2][i//2].axis('off')

      plt.savefig('/scratch/izar/parsaeif/lstm_res_visul/predicted_pose_e{:04d}-idx{:04d}-rnd{:04d}.png'.format(e,idx, rnd))



