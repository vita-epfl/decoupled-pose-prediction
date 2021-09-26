import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import utils
import model
import DataLoader_test
import argparse
import json

def main(args):
    
    #############################loading the data#####################################
    #dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = args.dev
    val = DataLoader_test.data_loader()
    ##################################################################################
    
    ##############################defining the model##################################
    net_g = model.LSTM_g(embedding_dim=args.embedding_dim, h_dim=args.hidden_dim, dropout=args.dropout, dev=dev).to(device=dev)
    encoder = model.Encoder(pose_dim=args.pose_dim, h_dim=args.hidden_dim, latent_dim=args.latent_dim, dev=dev)
    decoder = model.Decoder(pose_dim=args.pose_dim, h_dim=args.hidden_dim, latent_dim=args.latent_dim, dev=dev)
    net_l = model.VAE(Encoder=encoder, Decoder=decoder).to(device=dev)
    net_g.double()
    net_l.double()

    ########################load params#########################
    net_g.load_state_dict(torch.load("checkpoint_g.pkl"))
    net_l.load_state_dict(torch.load("checkpoint_l.pkl"))
    
    
    print('='*100)

    start = time.time()
    net_g.eval()
    net_l.eval()
    for idx, (obs_p, obs_s, obs_f, obs_m, target_f, start_end_idx) in enumerate(val):
    
        batch = obs_p.size(1)
        obs_p = obs_p.to(device=dev).double()
        obs_s = obs_s.to(device=dev).double()
        obs_m = obs_m.to(device=dev)
        
        with torch.no_grad():
    
            ####################predicting masks#######################
            preds_m = obs_m[-1].unsqueeze(0).repeat(14, 1, 1)

            #########splitting the motion into local + global##########
            obs_s_g = 0.5*(obs_s.view(15, batch, 14, 2)[:,:,8] + obs_s.view(15, batch, 14, 2)[:,:,9])
            obs_s_l = (obs_s.view(15, batch, 14, 2) - obs_s_g.view(15, batch, 1, 2)).view(15, batch, 28)
            ###########################################################
    
            #####predicting the global speed and calculate mse loss####
            speed_preds_g = net_g(global_s=obs_s_g)
            ######predicting the local speed using VAE and calculate loss########### 
            output, mean, log_var = net_l(obs_s_l)
            ###########################################################
            speed_preds = (speed_preds_g.view(14, batch, 1, 2) + output.view(14, batch, 14, 2)).view(14, batch, 28)
            preds_p = utils.speed2pos(speed_preds, obs_p, dev=dev)

            alist_p = []
            alist_m = []
            for _, (start, end) in enumerate(start_end_idx):
                alist_p.append(preds_p.permute(1,0,2)[start:end].tolist())
                alist_m.append(preds_m.permute(1,0,2)[start:end].tolist())

            with open('posetrack_predictions.json', 'w') as f:
                f.write(json.dumps(alist_p))
            with open('posetrack_masks.json', 'w') as f:
                f.write(json.dumps(alist_m))
    
    
    print('='*100) 
    print('Done !')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', default=64, type=int, required=False)
    parser.add_argument('--latent_dim', default=32, type=int, required=False)
    parser.add_argument('--embedding_dim', default=8, type=int, required=False)
    parser.add_argument('--pose_dim', default=28, type=int, required=False)
    parser.add_argument('--dropout', default=0.2, type=float, required=False)
    parser.add_argument('--dev', default="cpu", type=str, required=False)
    args = parser.parse_args()

    main(args)

