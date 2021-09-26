import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import utils
import time
import argparse
import model 
import jta_dataloader
import pdb

LEFT_HIP = 16
RIGHT_HIP = 19

def main(args):

    ######################loading data#######################
    args.dtype = "train"
    train = jta_dataloader.data_loader_JTA(args)
    args.dtype = 'validation'
    val = jta_dataloader.data_loader_JTA(args)
    args.dtype = 'val'


    ####################defining model#######################
    net_g = model.LSTM_g(embedding_dim=args.embedding_dim, D_dim=args.D_dim, h_dim=args.hidden_dim, dropout=args.dropout, pred_len=args.pred_len)
    encoder = model.Encoder(pose_dim=args.pose_dim, h_dim=args.hidden_dim, latent_dim=args.latent_dim, dropout=args.dropout)
    decoder = model.Decoder(pose_dim=args.pose_dim, h_dim=args.hidden_dim, latent_dim=args.latent_dim, dropout=args.dropout, pred_len=args.pred_len)
    net_l = model.VAE(Encoder=encoder, Decoder=decoder)
    if torch.cuda.is_available():
        net_l.cuda()
        net_g.cuda()
    net_l.double()
    net_g.double()
    net_params = list(net_l.parameters()) + list(net_g.parameters())

    ########################load params#########################
    if(args.load_checkpoint):
        net_g.load_state_dict(torch.load("checkpoint_g.pkl"))
        net_l.load_state_dict(torch.load("checkpoint_l.pkl"))

    #######defining loss_fn, optimizer, and scheduler########
    loss_fn = nn.MSELoss() 
    bce = nn.BCELoss()
    optimizer = optim.Adam(net_params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=7, threshold = 1e-8, verbose=True)
    
    print('='*100)
    print('Training ...')
    for epoch in range(args.n_epochs):
    
        start = time.time()
        ade_train  = 0
        fde_train  = 0
        vim_train = 0
        acc_train = 0
        loss_train = 0
        counter = 0
        net_g.train()
        net_l.train()
        #for idx, (obs_p, obs_s, obs_m, obs_f, target_p, target_s, target_m, target_f) in enumerate(train):
        for idx, (obs_p, obs_s, obs_f, obs_m, target_p, target_s, target_f, target_m, seq_start_end) in enumerate(train):
    
            batch = obs_p.size(1)
            counter += batch 
            if torch.cuda.is_available():
                obs_p = obs_p.to('cuda').double() 
                obs_s = obs_s.to('cuda').double()
                obs_m = obs_m.to('cuda').double()
                target_p = target_p.to('cuda').double() 
                target_s = target_s.to('cuda').double()
                target_m = target_m.to('cuda').double()

            #########splitting the motion into local + global##########
            obs_s_g = 0.5*(obs_s.view(args.obs_len-1, batch, args.pose_dim//args.D_dim, args.D_dim)[:,:,LEFT_HIP] + \
                           obs_s.view(args.obs_len-1, batch, args.pose_dim//args.D_dim, args.D_dim)[:,:,RIGHT_HIP])
            target_s_g = 0.5*(target_s.view(args.pred_len, batch, args.pose_dim//args.D_dim, args.D_dim)[:,:,LEFT_HIP] + \
                              target_s.view(args.pred_len, batch, args.pose_dim//args.D_dim, args.D_dim)[:,:,RIGHT_HIP])
            obs_s_l = (obs_s.view(args.obs_len-1, batch, args.pose_dim//args.D_dim, args.D_dim) - \
                       obs_s_g.view(args.obs_len-1, batch, 1, args.D_dim)).view(args.obs_len-1, batch, args.pose_dim)
            target_s_l = (target_s.view(args.pred_len, batch, args.pose_dim//args.D_dim, args.D_dim) - \
                          target_s_g.view(args.pred_len, batch, 1, args.D_dim)).view(args.pred_len, batch, args.pose_dim)
            ###########################################################
            net_g.zero_grad()
            #####predicting the global speed and calculate mse loss####
            speed_preds_g = net_g(global_s=obs_s_g)
            loss_g = loss_fn(speed_preds_g, target_s_g)
            ###########################################################
            net_l.zero_grad()
            ######predicting the local speed using VAE and calculate loss########### 
            output, mean, log_var = net_l(obs_s_l)
            loss_l = model.vae_loss_function(target_s_l, output, mean, log_var) #- 0.0001* torch.norm(output)

            preds_m = obs_m[-1].unsqueeze(0).repeat(args.pred_len, 1, 1)
            loss_m = bce(preds_m, target_m)*batch


            speed_preds = (speed_preds_g.view(args.pred_len, batch, 1, args.D_dim) + output.view(args.pred_len, batch, args.pose_dim//args.D_dim, args.D_dim)).view(args.pred_len, batch, args.pose_dim)
            preds_p = utils.speed2pos(speed_preds, obs_p) 
            #####################total loss############################
            loss = loss_g + 0.1*loss_l
            ####################backward and optimize##################
            loss.backward()
            ###########################################################
            optimizer.step()
            #####################calculating the metrics########################
            ade_train += float(utils.ADE_c(preds_p, target_p))
            fde_train += float(utils.FDE_c(preds_p, target_p))
            vim_train += utils.myVIM(preds_p, target_p)
            acc_train += torch.mean((preds_m==target_m).type(torch.float))*batch 
            loss_train += loss.item()*batch
    
        loss_train /= counter
        ade_train  /= counter
        fde_train  /= counter    
        vim_train  /= counter    
        acc_train /= counter
     
        ade_val  = 0
        fde_val  = 0
        vim_val = 0
        loss_val = 0
        acc_val = 0
        counter = 0
        net_g.eval()
        net_l.eval()
        #for idx, (obs_p, obs_s, obs_m, obs_f, target_p, target_s, target_m, target_f) in enumerate(val):
        for idx, (obs_p, obs_s, obs_f, obs_m, target_p, target_s, target_f, target_m, seq_start_end) in enumerate(val):
    
            batch = obs_p.size(1) 
            counter += batch 
            if torch.cuda.is_available():
                obs_p = obs_p.to('cuda').double() 
                obs_s = obs_s.to('cuda').double()
                obs_m = obs_m.to('cuda').double()
                target_p = target_p.to('cuda').double() 
                target_s = target_s.to('cuda').double()
                target_m = target_m.to('cuda').double()
            
            #########splitting the motion into local + global##########
            obs_s_g = 0.5*(obs_s.view(args.obs_len-1, batch, args.pose_dim//args.D_dim, args.D_dim)[:,:,LEFT_HIP] + \
                           obs_s.view(args.obs_len-1, batch, args.pose_dim//args.D_dim, args.D_dim)[:,:,RIGHT_HIP])
            target_s_g = 0.5*(target_s.view(args.pred_len, batch, args.pose_dim//args.D_dim, args.D_dim)[:,:,LEFT_HIP] + \
                              target_s.view(args.pred_len, batch, args.pose_dim//args.D_dim, args.D_dim)[:,:,RIGHT_HIP])
            obs_s_l = (obs_s.view(args.obs_len-1, batch, args.pose_dim//args.D_dim, args.D_dim) - \
                       obs_s_g.view(args.obs_len-1, batch, 1, args.D_dim)).view(args.obs_len-1, batch, args.pose_dim)
            target_s_l = (target_s.view(args.pred_len, batch, args.pose_dim//args.D_dim, args.D_dim) - \
                          target_s_g.view(args.pred_len, batch, 1, args.D_dim)).view(args.pred_len, batch, args.pose_dim)
            ###########################################################
            with torch.no_grad():
                #####predicting the global speed and calculate mse loss####
                speed_preds_g = net_g(global_s=obs_s_g)
                loss_g = loss_fn(speed_preds_g, target_s_g)
                ######predicting the local speed using VAE and calculate loss########### 
                output, mean, log_var = net_l(obs_s_l)
                loss_l = model.vae_loss_function(target_s_l, output, mean, log_var) #- 0.0001* torch.norm(output)
                ###########################################################
                speed_preds = (speed_preds_g.view(args.pred_len, batch, 1, args.D_dim) + output.view(args.pred_len, batch, args.pose_dim//args.D_dim, args.D_dim)).view(args.pred_len, batch, args.pose_dim)
                #####################total loss############################
                loss = loss_g + 0.1*loss_l

                ##################calculating the predictions#######################
                preds_p = utils.speed2pos(speed_preds, obs_p) 
                ##################calculating the intentions#######################
                preds_m = obs_m[-1].unsqueeze(0).repeat(args.pred_len, 1, 1)
                loss_m = bce(preds_m, target_m)*batch

                if(epoch %100 == 0): utils.visu(obs_p, obs_m, obs_f, preds_p, target_p, target_m, target_f, seq_start_end, "/work/vita/JTA_dataset/Original_JTA_dataset/frames/val/", epoch, idx)

            #####################calculating the metrics########################
            ade_val += float(utils.ADE_c(preds_p, target_p))
            fde_val += float(utils.FDE_c(preds_p, target_p))
            vim_val += utils.myVIM(preds_p, target_p)
            acc_val += torch.mean((preds_m==target_m).type(torch.float))*batch 
            loss_val += loss.item()*batch
    
        loss_val /= counter
        ade_val  /= counter
        fde_val  /= counter    
        vim_val  /= counter    
        acc_val /= counter
        scheduler.step(loss_val)
    
        print("e: %d "%epoch,
              "|loss_t: %0.4f "%loss_train,
              "|loss_v: %0.4f "%loss_val,
              "|fde_t: %0.4f "%fde_train,
              "|fde_v: %0.4f "%fde_val,
              "|ade_t: %0.4f "%ade_train, 
              "|ade_v: %0.4f "%ade_val, 
              "|vim_t: %0.4f "%vim_train,
              "|vim_v: %0.4f "%vim_val,
              "|acc_t: %0.4f "%acc_train,
              "|acc_v: %0.4f "%acc_val,
              "|time(s): %0.4f "%(time.time()-start)) 
    
    print('='*100) 
    print('Saving ...')
    torch.save(net_g.state_dict(), 'checkpoint_g.pkl')
    torch.save(net_l.state_dict(), 'checkpoint_l.pkl')

    print('Done !')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_dim', default=44, type=int, required=False)
    parser.add_argument('--D_dim', default=2, type=int, required=False)
    parser.add_argument('--obs_len', default=16, type=int, required=False)
    parser.add_argument('--pred_len', default=14, type=int, required=False)
    parser.add_argument('--stride', default=5, type=int, required=False)
    parser.add_argument('--hidden_dim', default=64, type=int, required=False)
    parser.add_argument('--latent_dim', default=32, type=int, required=False)
    parser.add_argument('--embedding_dim', default=16, type=int, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, required=False)
    parser.add_argument('--lr', default=0.01, type=float, required=False)
    parser.add_argument('--n_epochs', default=1000, type=int, required=False)
    parser.add_argument('--batch_size', default=64, type=int, required=False)
    parser.add_argument('--loader_shuffle', default=True, type=bool, required=False)
    parser.add_argument('--pin_memory', default=False, type=bool, required=False)
    parser.add_argument('--loader_workers', default=1, type=int, required=False)
    parser.add_argument('--load_checkpoint', default=False, type=bool, required=False)
    args = parser.parse_args()

    main(args)
