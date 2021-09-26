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
    args.dtype = "train"
    train = DataLoader.data_loader(args)
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
    net_params = list(net_l.parameters()) + list(net_g.parameters())
    ##################################################################################
    
    ##########################defining the optimizer##################################
    loss_fn = nn.MSELoss() 
    bce = nn.BCELoss()
    optimizer = optim.Adam(net_params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25, threshold = 1e-5, verbose=True)
    ##################################################################################
    
    print('='*100)
    print('Training ...')
    
    for epoch in range(args.n_epochs):

        start = time.time()
        counter = 0
        acc_train = 0
        ade_train  = 0
        fde_train  = 0
        vim_train = 0
        loss_train = 0
    
        net_g.train()
        net_l.train()
        for idx, (obs_p, obs_s, obs_f, obs_m, target_p, target_s, target_f, target_m, start_end_idx) in enumerate(train):
    
            batch = obs_p.size(1)
            counter += batch      
            obs_p = obs_p.to(device=dev).double()
            obs_s = obs_s.to(device=dev).double()
            target_p = target_p.to(device=dev).double()
            target_s = target_s.to(device=dev).double()
            obs_m = obs_m.to(device=dev)
            target_m = target_m.to(device=dev)
    
            #########splitting the motion into local + global##########
            obs_s_g = 0.5*(obs_s.view(15, batch, 14, 2)[:,:,8] + obs_s.view(15, batch, 14, 2)[:,:,9])
            target_s_g = 0.5*(target_s.view(14, batch, 14, 2)[:,:,8] + target_s.view(14, batch, 14, 2)[:,:,9])
            obs_s_l = (obs_s.view(15, batch, 14, 2) - obs_s_g.view(15, batch, 1, 2)).view(15, batch, 28)
            target_s_l = (target_s.view(14, batch, 14, 2) - target_s_g.view(14, batch, 1, 2)).view(14, batch, 28)
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

            ###########################################################
            speed_preds = (speed_preds_g.view(14, batch, 1, 2) + output.view(14, batch, 14, 2)).view(14, batch, 28)
            preds_p = utils.speed2pos(speed_preds, obs_p, dev=dev)
            #####################total loss############################
            loss = loss_g + 0.1*loss_l
            ####################backward and optimize##################
            loss.backward()
            ###########################################################
            optimizer.step()

            ####predicting masks and calculate bce loss for masks######
            predicted_masks = obs_m[-1].unsqueeze(0).repeat(14, 1, 1)
            loss_m = bce(predicted_masks, target_m)
            #####################calculating the metrics########################
            ade_train += float(utils.ADE_c(preds_p, target_p))
            fde_train += float(utils.FDE_c(preds_p, target_p))
            vim_train += utils.myVIM(preds_p, target_p, predicted_masks).item()
            acc_train += torch.mean((predicted_masks == target_m).type(torch.float)).item()*batch
            loss_train += loss.item()*batch 
            ####################################################################
    
        loss_train /= counter
        ade_train  /= counter
        fde_train  /= counter    
        vim_train  /= counter    
        acc_train /= counter
        scheduler.step(loss_train)
      
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
              "|time: %0.4f "%(time.time()-start))
    
    print('='*100) 
    print('Saving ...')
    torch.save(net_g.state_dict(), 'checkpoint_g.pkl')
    torch.save(net_l.state_dict(), 'checkpoint_l.pkl')
   
    print('Done !')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', default=64, type=int, required=False)
    parser.add_argument('--latent_dim', default=32, type=int, required=False)
    parser.add_argument('--embedding_dim', default=8, type=int, required=False)
    parser.add_argument('--pose_dim', default=28, type=int, required=False)
    parser.add_argument('--dropout', default=0.2, type=float, required=False)
    parser.add_argument('--lr', default=0.004, type=float, required=False)
    parser.add_argument('--n_epochs', default=100, type=int, required=False)
    parser.add_argument('--batch_size', default=60, type=int, required=False)
    parser.add_argument('--loader_shuffle', default=True, type=bool, required=False)
    parser.add_argument('--pin_memory', default=False, type=bool, required=False)
    parser.add_argument('--loader_workers', default=1, type=int, required=False)
    parser.add_argument('--load_checkpoint', default=False, type=bool, required=False)
    parser.add_argument('--dev', default="cpu", type=str, required=False)
    args = parser.parse_args()

    main(args)

