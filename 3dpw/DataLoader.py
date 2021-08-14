import torch
import os
import numpy as np
import json

class myJAAD(torch.utils.data.Dataset):
    def __init__(self, args):

        print('Loading', args.dtype, 'data ...')
        full_path = "./somof_data_3dpw/"

        self.args = args
        with open(os.path.join(full_path,"3dpw_"+args.dtype+"_frames_in.json"), 'r') as f:
            self.frames_in = json.load(f)

        with open(os.path.join(full_path,"3dpw_"+args.dtype+"_in.json"), 'r') as f:
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

        with open(os.path.join(full_path,"3dpw_"+args.dtype+"_out.json"), 'r') as f:
            self.data_out = json.load(f)

    def __len__(self):

        return len(self.data_in) 
    
    def __getitem__(self, idx):

        obs_p = torch.tensor(self.data_in[idx])   
        target_p = torch.tensor(self.data_out[idx]) 
        obs_s = (obs_p[:,1:,:] - obs_p[:,:-1,:]) 
        target_s = torch.cat(((target_p[:,0,:]-obs_p[:,-1,:]).unsqueeze(1), target_p[:,1:,:]-target_p[:,:-1,:]),dim=1)

        return obs_p, obs_s, self.frames_in[idx], target_p, target_s, self.frames_out[idx]

def my_collate(batch):

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
    obs_s = torch.cat(obs_s, dim=0).permute(1,0,2)
    target_p = torch.cat(target_p, dim=0).permute(1,0,2)
    target_s = torch.cat(target_s, dim=0).permute(1,0,2)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [obs_p, obs_s, obs_f, target_p, target_s, target_f, seq_start_end]

    return tuple(out)

    
def data_loader(args):

    dataset = myJAAD(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, collate_fn=my_collate, drop_last=False)

    return dataloader

