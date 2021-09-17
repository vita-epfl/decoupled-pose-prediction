import torch
import pandas as pd
from ast import literal_eval
import time
import os
import numpy as np


#PATH = "/work/vita/JTA_csvs/JTA/2D" 
PATH = "/home/parsaeif/posepred/preprocessed_data/JTA_interactive/2D"
class JTA_DataLoader(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        start = time.time()
        print(" Loading {} data ...".format(args.dtype))
        sequence_centric = pd.read_csv(os.path.join(PATH, args.dtype + "_{}_{}_{}_JTA.csv".format(args.obs_len, args.pred_len, args.stride))) #.tail(10000)
        df = sequence_centric.copy()
        for v in list(df.columns.values):
            print(f"   {v} loaded")
            try:
                df.loc[:, v] = df.loc[:, v].apply(lambda x: literal_eval(x))
            except:
                continue
        sequence_centric[df.columns] = df[df.columns]
        self.data = sequence_centric.copy().reset_index(drop=True)

        self.data = self.data.drop(self.data[self.data.observed_mask.apply(lambda x: 1 in x)].index)
        self.data = self.data.drop(self.data[self.data.future_mask.apply(lambda x: 1 in x)].index)
        self.data = self.data.reset_index(drop=True)
        
        self.data.to_csv(args.dtype + "_{}_{}_{}_JTA.csv".format(args.obs_len, args.pred_len, args.stride), index=False)
        print(f" data loaded in {time.time()-start} seconds")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data.iloc[index] 
        #observed_mask,future_mask,observed_image_path,future_image_path 
        obs = torch.tensor(seq.observed_pose)
        obs_speed = obs[:,1:] - obs[:,:-1]
        true = torch.tensor(seq.future_pose)
        true_speed = torch.cat(((true[:,0] - obs[:,-1]).unsqueeze(1), true[:,1:] - true[:,:-1]), dim=1)
        
        obs_m = torch.tensor(seq.observed_mask)
        true_m = torch.tensor(seq.future_mask)
        
        obs_f = seq.observed_image_path
        true_f = seq.future_image_path

        return obs, obs_speed, obs_m, obs_f, true, true_speed, true_m, true_f


#def my_collate(batch):
#    (obs_p, obs_s, obs_m, obs_f, target_p, target_s, target_m, target_f) = zip(*batch)
#    #print(f"in my_collate: obs_p {torch.stack(obs_p, dim=0).shape} target_p {torch.stack(target_p,dim=0).shape}")
#    obs_p = torch.stack(obs_p, dim=0).permute(1,0,2)
#    obs_s = torch.stack(obs_s, dim=0).permute(1,0,2)
#    obs_m = torch.stack(obs_m, dim=0).permute(1,0,2)
#    target_p = torch.stack(target_p, dim=0).permute(1,0,2)
#    target_s = torch.stack(target_s, dim=0).permute(1,0,2)
#    target_m = torch.stack(target_m, dim=0).permute(1,0,2)
#    out = [obs_p, obs_s, obs_m, obs_f, target_p, target_s, target_m, target_f]
#    return tuple(out)

def my_collate(batch):
    (obs_p, obs_s, obs_m, obs_f, target_p, target_s, target_m, target_f) = zip(*batch)
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
    target_p = torch.cat(target_p, dim=0).permute(1,0,2)
    target_s = torch.cat(target_s, dim=0).permute(1,0,2)
    target_m = torch.cat(target_m, dim=0).permute(1,0,2)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [obs_p, obs_s, obs_ff, obs_m, target_p, target_s, target_ff, target_m, seq_start_end]
    return tuple(out)



def data_loader_JTA(args):
    dataset = JTA_DataLoader(args)
  
    if(args.dtype == 'validation' or args.dtype == 'test'):
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=dataset.__len__(), shuffle=False,
            pin_memory=args.pin_memory, num_workers=args.loader_workers, collate_fn=my_collate, drop_last=False)
    elif(args.dtype == 'train'):
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
            pin_memory=args.pin_memory, num_workers=args.loader_workers, collate_fn=my_collate, drop_last=False)
    else:
        print(f"ERROR {args.dtype} not available!")
        exit()
    return dataloader


