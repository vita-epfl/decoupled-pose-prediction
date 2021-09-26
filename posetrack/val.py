import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import utils
import model
import DataLoader
import argparse

def main(args):
    
    #############################loading the data#####################################
    #dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = args.dev
    args.dtype = 'valid'
    val = DataLoader.data_loader(args)
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
    
    ##########################defining the optimizer##################################
    loss_fn = nn.MSELoss() 
    bce = nn.BCELoss()
    ##################################################################################
    
    print('='*100)

    start = time.time()
    counter = 0
    acc_val = 0
    ade_val  = 0
    fde_val  = 0
    vim_val = 0
    loss_val = 0
    
    net_g.eval()
    net_l.eval()
    for idx, (obs_p, obs_s, obs_f, obs_m, target_p, target_s, target_f, target_m, start_end_idx) in enumerate(val):
    
        batch = obs_p.size(1)
        counter += batch   
        obs_p = obs_p.to(device=dev).double()
        obs_s = obs_s.to(device=dev).double()
        target_p = target_p.to(device=dev).double()
        target_s = target_s.to(device=dev).double()
        obs_m = obs_m.to(device=dev)
        target_m = target_m.to(device=dev)
    
        
        with torch.no_grad():
    
            ####################predicting masks#######################
            predicted_masks = obs_m[-1].unsqueeze(0).repeat(14, 1, 1)
            loss_m = bce(predicted_masks, target_m)

            #########splitting the motion into local + global##########
            obs_s_g = 0.5*(obs_s.view(15, batch, 14, 2)[:,:,8] + obs_s.view(15, batch, 14, 2)[:,:,9])
            target_s_g = 0.5*(target_s.view(14, batch, 14, 2)[:,:,8] + target_s.view(14, batch, 14, 2)[:,:,9])
            obs_s_l = (obs_s.view(15, batch, 14, 2) - obs_s_g.view(15, batch, 1, 2)).view(15, batch, 28)
            target_s_l = (target_s.view(14, batch, 14, 2) - target_s_g.view(14, batch, 1, 2)).view(14, batch, 28)
            ###########################################################
    
            #####predicting the global speed and calculate mse loss####
            speed_preds_g = net_g(global_s=obs_s_g)
            loss_g = loss_fn(speed_preds_g, target_s_g)
            ######predicting the local speed using VAE and calculate loss########### 
            output, mean, log_var = net_l(obs_s_l)
            loss_l = model.vae_loss_function(target_s_l, output, mean, log_var) #- 0.0001* torch.norm(output)
            ###########################################################
            speed_preds = (speed_preds_g.view(14, batch, 1, 2) + output.view(14, batch, 14, 2)).view(14, batch, 28)
            preds_p = utils.speed2pos(speed_preds, obs_p, dev=dev)
            #####################total loss############################
            loss = loss_g + 0.1*loss_l
    
            #####################calculating the metrics########################
            ade_val += float(utils.ADE_c(preds_p, target_p))
            fde_val += float(utils.FDE_c(preds_p, target_p))
            vim_val += utils.myVIM(preds_p, target_p, predicted_masks)
            loss_val += loss.item()*batch 
            acc_val += torch.mean((predicted_masks == target_m).type(torch.float)).item()*batch
    
    loss_val /= counter
    ade_val  /= counter
    fde_val  /= counter    
    vim_val  /= counter    
    acc_val /= counter
    
    print("|loss_v: %0.4f "%loss_val,
          "|fde_v: %0.4f "%fde_val,
          "|ade_v: %0.4f "%ade_val, 
          "|vim_v: %0.4f "%vim_val, 
          "|acc_v: %0.4f "%acc_val,
          "|time: %0.4f "%(time.time()-start))
    
    print('='*100) 
    print('Done !')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', default=64, type=int, required=False)
    parser.add_argument('--latent_dim', default=32, type=int, required=False)
    parser.add_argument('--embedding_dim', default=8, type=int, required=False)
    parser.add_argument('--pose_dim', default=28, type=int, required=False)
    parser.add_argument('--dropout', default=0.2, type=float, required=False)
    parser.add_argument('--batch_size', default=60, type=int, required=False)
    parser.add_argument('--loader_shuffle', default=True, type=bool, required=False)
    parser.add_argument('--pin_memory', default=False, type=bool, required=False)
    parser.add_argument('--loader_workers', default=1, type=int, required=False)
    parser.add_argument('--load_checkpoint', default=False, type=bool, required=False)
    parser.add_argument('--dev', default="cpu", type=str, required=False)
    args = parser.parse_args()

    main(args)

