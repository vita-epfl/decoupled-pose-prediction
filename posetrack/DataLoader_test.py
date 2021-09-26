import torch
import os
import numpy as np
import json

class myJAAD(torch.utils.data.Dataset):
    def __init__(self):
        print('Loading', 'test', 'data ...')
        full_path = "./somof_data_posetrack/"

        #image_size: 1280, 720
        with open(os.path.join(full_path,"posetrack_test_frames_in.json"), 'r') as f:
            self.frames_in = json.load(f)
        #(306, 16)
        with open(os.path.join(full_path,"posetrack_test_masks_in.json"), 'r') as f:
            self.masks_in = json.load(f)

        with open(os.path.join(full_path,"posetrack_test_in.json"), 'r') as f:
            self.data_in = json.load(f)

        frames_out = []
        for i in range(len(self.frames_in)):
           frames_out.append([])
           path = "/".join(self.frames_in[i][-1].split("/")[:-1])
           last = int(self.frames_in[i][-1].split("/")[-1].split(".")[0]) + 1
           for j in range(14):
              frames_out[i].append(path+"/{:06d}.jpg".format(last+j))
        self.frames_out = frames_out

    def __len__(self):
        return len(self.data_in) 
    
    def __getitem__(self, idx):
                #preprocessing
        obs_p = torch.tensor(self.data_in[idx])
        obs_m = torch.tensor(self.masks_in[idx])
        obs_p[ torch.repeat_interleave(obs_m, 2, dim=-1) == 0 ] = np.NaN
        obs_s = obs_p[:,1:,:] - obs_p[:,:-1,:]
        n_ped, seq_len, l = obs_s.shape
        _ = torch.ones(n_ped, seq_len, l//2)
        _[torch.isnan(obs_s[:,:,::2])] = 0
        velocity_obs_m = _

        obs_s[torch.isnan(obs_s)] = 0
        obs_p[torch.isnan(obs_p)] = 0
        ###################

        return obs_p, obs_s, self.frames_in[idx], obs_m, self.frames_out[idx]


#    def __getitem__(self, idx):
#
#        obs_keypoints = torch.tensor(self.data_in[idx])   #n_ped,16,28
#        obs_speed_keypoints = (obs_keypoints[:,1:,:] - obs_keypoints[:,:-1,:]) #n_ped, 15, 28
#
#        return obs_keypoints, obs_speed_keypoints, self.frames_in[idx], torch.tensor(self.masks_in[idx]), self.frames_out[idx]

def my_collate(batch):

    (obs_p, obs_s, obs_f, obs_m, target_f) = zip(*batch)
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
    obs_m = torch.cat(obs_m, dim=0).permute(1,0,2)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [obs_p, obs_s, obs_ff, obs_m, target_ff, seq_start_end]
    return tuple(out)

    
def data_loader():
    dataset = myJAAD()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=dataset.__len__(), shuffle=False, collate_fn=my_collate)

    return dataloader

