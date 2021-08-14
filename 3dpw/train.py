import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import utils
import time
import argparse
import model 
import DataLoader

def main(args):

    ######################loading data#######################
    args.dtype = "train"
    train = DataLoader.data_loader(args)
    args.dtype = 'valid'
    val = DataLoader.data_loader(args)
    
    ####################defining model#######################
    net_g = model.LSTM_g(embedding_dim=args.embedding_dim, h_dim=args.hidden_dim, dropout=args.dropout)
    encoder = model.Encoder(h_dim=args.hidden_dim, latent_dim=args.latent_dim, dropout=args.dropout)
    decoder = model.Decoder(h_dim=args.hidden_dim, latent_dim=args.latent_dim, dropout=args.dropout)
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
    optimizer = optim.Adam(net_params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=35, threshold = 1e-8, verbose=True)
    
    print('='*100)
    print('Training ...')
    for epoch in range(args.n_epochs):
    
        start = time.time()
        ade_train  = 0
        fde_train  = 0
        vim_train = 0
        loss_train = 0
        counter = 0
        net_g.train()
        net_l.train()
        for idx, (obs_p, obs_s, obs_f, target_p, target_s, target_f, start_end_idx) in enumerate(train):
    
            batch = obs_p.size(1)
            counter += batch 
            obs_p = obs_p.to(device='cpu').double()
            obs_s = obs_s.to(device='cpu').double()
            target_p = target_p.to(device='cpu').double()
            target_s = target_s.to(device='cpu').double()
    
            #########splitting the motion into local + global##########
            obs_s_g = 0.5*(obs_s.view(15, batch, 13, 3)[:,:,0] + obs_s.view(15, batch, 13, 3)[:,:,1])
            target_s_g = 0.5*(target_s.view(14, batch, 13, 3)[:,:,0] + target_s.view(14, batch, 13, 3)[:,:,1])
            obs_s_l = (obs_s.view(15, batch, 13, 3) - obs_s_g.view(15, batch, 1, 3)).view(15, batch, 39)
            target_s_l = (target_s.view(14, batch, 13, 3) - target_s_g.view(14, batch, 1, 3)).view(14, batch, 39)
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
            speed_preds = (speed_preds_g.view(14, batch, 1, 3) + output.view(14, batch, 13, 3)).view(14, batch, 39)
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
            loss_train += loss.item()*batch
    
        loss_train /= counter
        ade_train  /= counter
        fde_train  /= counter    
        vim_train  /= counter    
        scheduler.step(loss_train)
     
        ade_val  = 0
        fde_val  = 0
        vim_val = 0
        loss_val = 0
        counter = 0
        net_g.eval()
        net_l.eval()
        for idx, (obs_p, obs_s, obs_f, target_p, target_s, target_f, start_end_idx) in enumerate(val):
    
            batch = obs_p.size(1) 
            counter += batch 
            obs_p = obs_p.to(device='cpu').double()
            obs_s = obs_s.to(device='cpu').double()
            target_p = target_p.to(device='cpu').double()
            target_s = target_s.to(device='cpu').double()
            
            #########splitting the motion into local + global##########
            obs_s_g = 0.5*(obs_s.view(15, batch, 13, 3)[:,:,0] + obs_s.view(15, batch, 13, 3)[:,:,1])
            target_s_g = 0.5*(target_s.view(14, batch, 13, 3)[:,:,0] + target_s.view(14, batch, 13, 3)[:,:,1])
            obs_s_l = (obs_s.view(15, batch, 13, 3) - obs_s_g.view(15, batch, 1, 3)).view(15, batch, 39)
            target_s_l = (target_s.view(14, batch, 13, 3) - target_s_g.view(14, batch, 1, 3)).view(14, batch, 39)
            ###########################################################
            with torch.no_grad():
                #####predicting the global speed and calculate mse loss####
                speed_preds_g = net_g(global_s=obs_s_g)
                loss_g = loss_fn(speed_preds_g, target_s_g)
                ######predicting the local speed using VAE and calculate loss########### 
                output, mean, log_var = net_l(obs_s_l)
                loss_l = model.vae_loss_function(target_s_l, output, mean, log_var) #- 0.0001* torch.norm(output)
                ###########################################################
                speed_preds = (speed_preds_g.view(14, batch, 1, 3) + output.view(14, batch, 13, 3)).view(14, batch, 39)
                #####################total loss############################
                loss = loss_g + 0.1*loss_l

                ##################calculating the predictions#######################
                preds_p = utils.speed2pos(speed_preds, obs_p) 

            #####################calculating the metrics########################
            ade_val += float(utils.ADE_c(preds_p, target_p))
            fde_val += float(utils.FDE_c(preds_p, target_p))
            vim_val += utils.myVIM(preds_p, target_p)
            loss_val += loss.item()*batch
    
        loss_val /= counter
        ade_val  /= counter
        fde_val  /= counter    
        vim_val  /= counter    
    
        print("e: %d "%epoch,
              "|loss_t: %0.6f "%loss_train,
              "|loss_v: %0.6f "%loss_val,
              "|fde_t: %0.6f "%fde_train,
              "|fde_v: %0.6f "%fde_val,
              "|ade_t: %0.6f "%ade_train, 
              "|ade_v: %0.6f "%ade_val, 
              "|vim_t: %0.6f "%vim_train,
              "|vim_v: %0.6f "%vim_val,
              "|time(s): %0.6f "%(time.time()-start)) 
    
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
    parser.add_argument('--dropout', default=0.2, type=float, required=False)
    parser.add_argument('--lr', default=0.004, type=float, required=False)
    parser.add_argument('--n_epochs', default=100, type=int, required=False)
    parser.add_argument('--batch_size', default=60, type=int, required=False)
    parser.add_argument('--loader_shuffle', default=True, type=bool, required=False)
    parser.add_argument('--pin_memory', default=False, type=bool, required=False)
    parser.add_argument('--loader_workers', default=1, type=int, required=False)
    parser.add_argument('--load_checkpoint', default=False, type=bool, required=False)
    args = parser.parse_args()

    main(args)

