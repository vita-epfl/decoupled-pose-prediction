import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import time

from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score

import DataLoader


device = 'cuda'
import utils

from PIL import Image, ImageDraw
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import args_module
scale =  args_module.args().scale
scalex = args_module.args().scalex
scaley = args_module.args().scaley
import random
random.seed(10)
torch.manual_seed(10)
import numpy as np
np.random.seed(0)

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
      
      plt.savefig('/scratch/izar/parsaeif/gan_res_visul/obs_pose_e{:04d}-idx{:04d}-rnd{:04d}.png'.format(e, idx, rnd))

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
              if(target_masks is not None):
                 plot_joints(img, target_p[rnd][3*i+1], target_masks[rnd][3*i+1], scale, scalex, scaley, color='green', width=width)
              #plot_keypoints(img, obs_pose[3*i+1], color='green')
              ax.imshow(image)
          #ax[i%2][i//2].axis('off')

      plt.savefig('/scratch/izar/parsaeif/gan_res_visul/predicted_pose_e{:04d}-idx{:04d}-rnd{:04d}.png'.format(e,idx, rnd))



#class Generator(nn.Module):
#    '''
#    Generator Class
#    Values:
#        z_dim: the dimension of the noise vector, a scalar
#        pose_dim: the dimension of poses 
#        hidden_dim: the inner dimension, a scalar
#    '''
#    def __init__(self, z_dim=128, pose_dim=34, hidden_dim=128):
#        super(Generator, self).__init__()
#        # Build the neural network
#        self.pose_encoder = nn.LSTM(input_size=pose_dim, hidden_size=hidden_dim)
#        self.pose_decoder = nn.LSTMCell(input_size=pose_dim, hidden_size=hidden_dim+z_dim)
#        self.fc_pose = nn.Linear(in_features=hidden_dim+z_dim, out_features=pose_dim)
#        self.tanh = nn.Tanh()
#        
#    def forward(self, noise, pose):
#        '''
#        Function for completing a forward pass of the generator: Given a noise tensor, 
#        returns generated poses.
#        Parameters:
#            noise: a noise tensor with dimensions (n_batch, z_dim)
#        '''
#
#        _, (hpo, cpo) = self.pose_encoder(pose.permute(1,0,2))
#        hpo = hpo.squeeze(0)
#        cpo = cpo.squeeze(0)
#
#        pose_outputs = torch.tensor([], device=device)
#        in_p = pose[:,-1,:]
#        
#        hdp = torch.cat((hpo, noise), dim=-1) 
#        cdp = torch.cat((cpo, noise), dim=-1) 
#        for i in range(16):
#            hdp, cdp = self.pose_decoder(in_p, (hdp, cdp))
#            pose_output = self.tanh(self.fc_pose(hdp))
#            pose_outputs = torch.cat((pose_outputs, pose_output.unsqueeze(1)), dim = 1)
#            in_p = pose_output.detach()
#            
#        return pose_outputs 

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        pose_dim: the dimension of poses 
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=128, pose_dim=34, hidden_dim=128):
        super(Generator, self).__init__()
        # Build the neural network
        self.pose_encoder = nn.LSTM(input_size=pose_dim, hidden_size=hidden_dim)

        self.speed_encoder = nn.LSTM(input_size=pose_dim, hidden_size=hidden_dim)
        self.speed_decoder = nn.LSTMCell(input_size=pose_dim, hidden_size=2*hidden_dim+z_dim)
        self.fc_speed = nn.Linear(in_features=2*hidden_dim+z_dim, out_features=pose_dim)

        self.mask_encoder = nn.LSTM(input_size=pose_dim//2, hidden_size=hidden_dim)
        self.mask_decoder = nn.LSTMCell(input_size=pose_dim//2, hidden_size=hidden_dim)
        self.fc_mask = nn.Linear(in_features=hidden_dim, out_features=pose_dim//2)
        
        self.tanh = nn.Tanh()
        self.hardtanh = nn.Hardtanh(-100, 100)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise, pose, speed, mask):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated poses.
        Parameters:
            noise: a noise tensor with dimensions (n_batch, z_dim)
        '''
        _, (hpo, cpo) = self.pose_encoder(pose.permute(1,0,2))
        hpo = hpo.squeeze(0)
        cpo = cpo.squeeze(0)

        _, (hso, cso) = self.speed_encoder(speed.permute(1,0,2))
        hso = hso.squeeze(0)
        cso = cso.squeeze(0)

        _, (hsm, csm) = self.mask_encoder(mask.permute(1,0,2))
        hsm = hsm.squeeze(0)
        csm = csm.squeeze(0)

        hdp = torch.cat((hpo, hso, noise), dim=-1)
        cdp = torch.cat((cpo, cso, noise), dim=-1)

        speed_outputs = torch.tensor([], device=device)
        in_sp = speed[:,-1,:]

        for i in range(14):
            hdp, cdp = self.speed_decoder(in_sp, (hdp, cdp))
            pose_output = self.tanh(self.fc_pose(hdp))
            #speed_output = self.hardtanh(self.fc_speed(hdp))
            speed_outputs = torch.cat((speed_outputs, speed_output.unsqueeze(1)), dim = 1)
            in_sp = speed_output #.detach()

        mask_outputs = torch.tensor([], device=device)
        in_m = mask[:,-1,:]

        for i in range(14):
            hsm, csm = self.mask_decoder(in_m, (hsm, csm))
            mask_output = self.sigmoid(self.fc_mask(hsm))
            mask_outputs = torch.cat((mask_outputs, mask_output.unsqueeze(1)), dim = 1)
            in_m = mask_output #.detach()

        return speed_outputs, mask_outputs

    
class MLPGenerator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        pose_dim: the dimension of poses 
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, seq_in, seq_out, pose_dim, noise_dim=128, hidden_dim=128):
        super(MLPGenerator, self).__init__()

        # Build the MLP neural network
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.pose_dim = pose_dim
        self.noise_dim = noise_dim

        self.gen = nn.Sequential(
                nn.Linear(pose_dim*seq_in+noise_dim, hidden_dim), 
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2*hidden_dim),
                nn.BatchNorm1d(2*hidden_dim),
                nn.ReLU(),
                nn.Linear(2*hidden_dim, 4*hidden_dim), 
                nn.BatchNorm1d(4*hidden_dim),
                nn.ReLU(),
                nn.Linear(4*hidden_dim, pose_dim*seq_out),
                )
        
    def forward(self, noise, pose, nothing):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated poses.
        Parameters:
            noise: a noise tensor with dimensions (n_batch, z_dim)
        '''
        #pose => (n_batch, seq_in, pose_dim)
        n_batch, seq_in, pose_dim = pose.shape
        assert(seq_in== self.seq_in and pose_dim==self.pose_dim)
        assert(noise.shape==(n_batch, self.noise_dim))
        
        #torch.cat((pose.view(n_batch, -1), noise), dim=-1) => (n_batch, noise_dim+seq_in*pose_dim)
        pose_out = self.gen(torch.cat((pose.view(n_batch, -1), noise), dim=-1)).view(n_batch, self.seq_out, pose_dim) #=> (n_batch, seq_out*pose_dim)
        return pose_out

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        pose_dim: the dimension of the poses
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, pose_dim=34, hidden_dim=128):
        super(Discriminator, self).__init__()

        self.speed_LSTM = nn.LSTM(input_size=pose_dim, hidden_size=hidden_dim)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=1)
        self.tanh = nn.Tanh()

    def forward(self, speed):
        '''
        Function for completing a forward pass of the discriminator: Given a pose, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            poses:  (pose_dim)
        '''
        _ , (hpo, cpo) = self.speed_LSTM(speed.permute(1,0,2)) #_ => (seq_len, batch_size, hidden_dim) 
        _ = _.permute(1,0,2) #=> (batch_size, seq_len, hidden_dim)
        #_ = self.tanh(self.linear(_)) # _ => (batch_size, seq_len, 1) #16 => should be (batch_size, 1)
        _ = self.linear(_) # _ => (batch_size, seq_len, 1) #16 => should be (batch_size, 1)
        

        return _


class MLPDiscriminator(nn.Module):
    '''
    MLPDiscriminator Class
    Values:
        pose_dim: the dimension of the poses
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, seq, pose_dim, hidden_dim=128):
        super(MLPDiscriminator, self).__init__()

        self.seq = seq
        self.pose_dim = pose_dim
        self.disc = nn.Sequential(
                nn.Linear(pose_dim*seq, hidden_dim), 
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, 2*hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(2*hidden_dim, 4*hidden_dim), 
                nn.LeakyReLU(0.2),
                nn.Linear(4*hidden_dim, 1),
                )

    def forward(self, pose):
        '''
        Function for completing a forward pass of the discriminator: Given a pose, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            poses:  (pose_dim)
        '''
        #pose => (n_batch, seq, pose_dim)
        n_batch, seq, pose_dim = pose.shape
        assert(seq== self.seq and pose_dim==self.pose_dim)
        
        out = self.disc(pose.view(n_batch, -1)) #=> (n_batch, seq_out*pose_dim)
        return out

    

#def get_disc_loss(gen, disc, criterion, pose_obs, pose_target, batch_size, noise, device):
#    '''
#    Return the loss of the discriminator given inputs.
#    Parameters:
#        gen: the generator model, which returns a seq of poses given z-dimensional noise
#        disc: the discriminator model, which returns a seq of single-dimensional prediction of real/fake
#        criterion: the loss function, which should be used to compare 
#               the discriminator's predictions to the ground truth reality of the images 
#               (e.g. fake = 0, real = 1)
#        pose_target: a batch of real target poses
#        pose_obs: a batch of observed poses
#        batch_size: the number of poses the generator should produce, 
#                which is also the length of the real images
#        z_dim: the dimension of the noise vector, a scalar
#        device: the device type
#    Returns:
#        disc_loss: a torch scalar loss value for the current batch
#    '''
#
#    #noise = torch.randn(batch_size, z_dim, device=device)
#    fake_target_poses = gen(noise, pose_obs)
#    
#    disc_fake_pred = disc(fake_target_poses.detach())
#    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
#    
#    disc_real_pred = disc(pose_target)
#    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
#    
#    disc_loss = (disc_real_loss + disc_fake_loss)/2
#    return disc_loss

def get_disc_loss(gen, disc, criterion, pose_obs, speed_obs, obs_masks, speed_target, batch_size, noise, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns a seq of poses given z-dimensional noise
        disc: the discriminator model, which returns a seq of single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        pose_target: a batch of real target poses
        pose_obs: a batch of observed poses
        batch_size: the number of poses the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''

    #noise = torch.randn(batch_size, z_dim, device=device)
    fake_target_speed, _ = gen(noise, pose_obs, speed_obs, obs_masks)

    disc_fake_pred = disc(fake_target_speed.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

    disc_real_pred = disc(speed_target)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

    disc_loss = (disc_real_loss + disc_fake_loss)/2
    return disc_loss


def get_gen_loss(gen, disc, criterion, batch_size, pose_obs, speed_obs, obs_masks, noise, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        batch_size: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    #noise = torch.randn(batch_size, z_dim, device=device)
    fake_target_speed, _ = gen(noise, pose_obs, speed_obs, obs_masks)
    
    disc_fake_pred = disc(fake_target_speed)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

    return gen_loss


'''
   input: observed body poses
   output: predicted body poses 
'''

z_dim = 128
args = args_module.args()


#gen = MLPGenerator(seq_in=args.input, seq_out=args.output, pose_dim=34, noise_dim=128, hidden_dim=256).to(args.device)
gen = Generator(z_dim=128, pose_dim=28, hidden_dim=256).to(args.device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=args.lr)
#disc = MLPDiscriminator(seq=args.output, pose_dim=34, hidden_dim=128).to(args.device) 
disc = Discriminator(pose_dim=28, hidden_dim=256).to(args.device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=args.lr)

#gen.load_state_dict(torch.load('generator.pkl'))
#disc.load_state_dict(torch.load('discriminator.pkl'))

train = DataLoader.data_loader(args)
args.dtype = 'valid'
args.save_path = args.save_path.replace('train', 'val')
args.file = args.file.replace('train', 'val')
val = DataLoader.data_loader(args)

scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(gen_opt, factor=0.5, patience=60, 
                                                 threshold = 1e-7, verbose=True)
scheduler_disc = optim.lr_scheduler.ReduceLROnPlateau(disc_opt, factor=0.5, patience=30, 
                                                 threshold = 1e-7, verbose=True)
#mse = nn.MSELoss()
mse = nn.L1Loss()
bce = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

best_model_val = 100000.
print('='*100)
print('Training ...')
for epoch in range(args.n_epochs):
    start = time.time()
    
    mean_discriminator_loss_train = 0.
    mean_generator_loss_train = 0.
    mean_p_loss_train = 0.
    mean_v_loss_train = 0.
    epoch_acc_train = 0. 
    fde_train = 0.
    ade_train = 0. 
    vim_train = 0. 
    
    counter = 0
    Counter = 0
    for idx, (obs_kp, obs_skp, obs_frames, obs_masks, target_kp, target_skp, target_frames, future_masks) in enumerate(train):

        counter += 1
        obs_kp = obs_kp.to(device=args.device)
        obs_skp = obs_skp.to(device=args.device)
        target_kp = target_kp.to(device=args.device)
        target_skp = target_skp.to(device=args.device)
        obs_masks = obs_masks.to(device=args.device)
        future_masks = future_masks.to(device=args.device)


        if(idx%1 == 0):
            disc_opt.zero_grad()
            noise = torch.randn(args.batch_size, z_dim, device=args.device)
            disc_loss = get_disc_loss(gen, disc, criterion, obs_kp, obs_skp, obs_masks, target_skp, args.batch_size, noise, args.device)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()
 
        gen_opt.zero_grad()
        noise = torch.randn(args.batch_size, z_dim, device=args.device)
        gen_loss = get_gen_loss(gen, disc, criterion, args.batch_size, obs_kp, obs_skp, obs_masks, noise, args.device)
        mean_generator_loss_train += gen_loss.item() 
        speed_preds, predicted_masks = gen(noise, obs_kp, obs_skp, obs_masks)
        pose_preds = utils.speed2bodypos(speed_preds, obs_kp)
        fde_train += utils.FDE_keypoints(pose_preds, target_kp)
        ade_train += utils.ADE_keypoints(pose_preds, target_kp)
       
        p_loss = mse(speed_preds, target_skp) #this term
        v_loss = bce(predicted_masks, future_masks)
        #print(f"ge_loss {gen_loss} 100*p_loss {100*p_loss} v_loss {v_loss} ")
        gen_loss += 200.*p_loss + v_loss
        #gen_loss += mse(pose_preds, target_kp) #this term
        #gen_loss += bce(predicted_masks, future_masks)
        mean_p_loss_train += p_loss 
        mean_v_loss_train += v_loss 
        # Update gradients
        gen_loss.backward(retain_graph=True)

        # Update optimizer
        gen_opt.step()
        scheduler_disc.step(gen_loss)
        if(idx%1 == 0):
             pass
             #scheduler_disc.step(disc_loss)
        
        # Keep track of the average discriminator loss
        mean_discriminator_loss_train += disc_loss.item() 

        # Keep track of the average generator loss
 
        #noise = torch.randn(args.batch_size, z_dim, device=device)
        predicted_masks = torch.where(predicted_masks>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))
        epoch_acc_train += sum(predicted_masks.view(-1)==future_masks.view(-1))
        #Counter += predicted_masks.view(-1).shape[0]
        Counter += predicted_masks.shape[0]
        vim_train += utils.myVIM(pose_preds, target_kp, predicted_masks)


    
       
    mean_discriminator_loss_train /= counter
    mean_generator_loss_train /= counter
    mean_p_loss_train /= counter
    mean_v_loss_train /= counter 
    fde_train /= counter
    ade_train /= counter
    vim_train /= (Counter*args.output)
    epoch_acc_train /= (Counter*args.output*14)
  

    fde_val = 0. 
    ade_val = 0.
    vim_val = 0.
    epoch_acc_val = 0. 

    counter = 0
    Counter = 0
    for idx, (obs_kp, obs_skp, obs_frames, obs_masks, target_kp, target_skp, target_frames, future_masks) in enumerate(val):

        counter += 1
        obs_kp = obs_kp.to(device=args.device)
        obs_skp = obs_skp.to(device=args.device)
        target_kp = target_kp.to(device=args.device)
        target_skp = target_skp.to(device=args.device)
        obs_masks = obs_masks.to(device=args.device)
        future_masks = future_masks.to(device=args.device)

        with torch.no_grad():
            
            noise = torch.randn(args.batch_size, z_dim, device=args.device)
            speed_preds, predicted_masks = gen(noise, obs_kp, obs_skp, obs_masks)
            pose_preds = utils.speed2bodypos(speed_preds, obs_kp)
            fde_val += utils.FDE_keypoints(pose_preds, target_kp)
            ade_val += utils.ADE_keypoints(pose_preds, target_kp)
            predicted_masks = torch.where(predicted_masks>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))
            epoch_acc_val += sum(predicted_masks.view(-1)==future_masks.view(-1))
            #Counter += predicted_masks.view(-1).shape[0]
            Counter += predicted_masks.shape[0]
            vim_val += utils.myVIM(pose_preds, target_kp, predicted_masks)

    fde_val /= counter
    ade_val /= counter
    vim_val /= (Counter*args.output)
    epoch_acc_val /= (Counter*args.output*14)
   
    #rnd = 0 
    g = torch.Generator()
    g.manual_seed(epoch%10)
    rnd = torch.randint(low=0, high=obs_kp.shape[0]-1, size=(1,), generator=g).item()

    #rnd = torch.randint(low=0, high=obs_kp.shape[0]-1, size=(1,)).item()
    plotter(obs_frames, obs_kp, obs_masks, target_frames, pose_preds, predicted_masks, target_kp, future_masks, epoch, idx, rnd, width=10)
    
    #plotter(obs_frames, obs_p, obs_masks, target_frames, preds_p, preds_masks, target_kp, future_masks, e, idx, rnd, width=10)


    print("e: %d "%epoch,
          "|gen_loss: %0.4f "%mean_generator_loss_train,
          "|disc_loss: %0.4f "%mean_discriminator_loss_train,
          "|p_loss: %0.4f "%mean_p_loss_train,
          "|v_loss: %0.4f "%mean_v_loss_train,
          "|acc: %0.4f "%epoch_acc_train, 
          "|vim: %0.4f "%vim_train, 
          "|fde: %0.4f "%fde_train, 
          "|ade: %0.4f "%ade_train, 
          "|acc: %0.4f "%epoch_acc_val, 
          "|vim: %0.4f "%vim_val, 
          "|fde_val: %0.4f "%fde_val, 
          "|ade_val: %0.4f "%ade_val) 
         
    #if(False and vim_val < best_model_val):
    if(vim_val < best_model_val):
         best_model_val = vim_val
         torch.save(gen.state_dict(), 'models/generator_vel.pkl')
         torch.save(disc.state_dict(), 'models/discriminator_vel.pkl')
         ################### generate predictions ##############

         # load test data
         import DataLoader_test
         args.dtype = 'test'
         test, info = DataLoader_test.data_loader(args)
         obs_kp, obs_skp, obs_frames, obs_masks, future_frames = next(iter(test)) 
         # move data to device
         obs_kp = obs_kp.to(device=args.device)
         obs_skp = obs_skp.to(device=args.device)
         obs_masks = obs_masks.to(device=args.device)
         print("deb: mask shape: ", obs_masks.shape)

         with torch.no_grad():
             noise = torch.randn(obs_kp.shape[0], z_dim, device=args.device)
             speed_preds, predicted_masks = gen(noise, obs_kp, obs_skp, obs_masks)
             pose_preds = utils.speed2bodypos(speed_preds, obs_kp)

         predicted_masks = torch.where(predicted_masks>=0.5, torch.tensor([1],device=args.device), torch.tensor([0], device=args.device))
         n_batch, seq_len, l = pose_preds.shape
         assert(l == 28)
         pose_preds = pose_preds.view(n_batch, seq_len, 14, 2)
         pose_preds[:,:,:,0] *= 1280
         pose_preds[:,:,:,1] *= 720
         pose_preds = pose_preds.view(n_batch, seq_len, 28)

         # collect pedestrians
         lst = []
         lst_ = []
         count = -1
         for v_idx in range(len(info)):
            lst.append([])
            lst_.append([])
            for p_idx in range(info[v_idx]):
                count += 1
                lst[v_idx].append( pose_preds[count].tolist() )
                lst_[v_idx].append( predicted_masks[count].tolist() )

         # write it to file
         import json
         alst = json.dumps(lst)
         alst_ = json.dumps(lst_)
         with open('gan_vel/posetrack_predictions.json', 'w') as f:
              f.write(alst)
         with open('gan_vel/posetrack_masks.json', 'w') as f:
              f.write(alst_)



         

    #if(epoch % 10 == 0 and epoch != 0):
    #   print('Saving epoch {}'.format(epoch))
    #   torch.save(gen.state_dict(), 'models/generator.pkl')
    #   torch.save(disc.state_dict(), 'models/discriminator.pkl')

print('='*100) 
#print('Saving ...')
#torch.save(gen.state_dict(), 'models/generator.pkl')
#torch.save(disc.state_dict(), 'models/discriminator.pkl')
print('Done !')

    


