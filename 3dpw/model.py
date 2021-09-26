import torch
import torch.nn as nn
import numpy as np

class LSTM_g(nn.Module):
    def __init__(self, pose_dim=39, embedding_dim=8, h_dim=16, num_layers=1, dropout=0.2, dev='cpu'):

        super(LSTM_g, self).__init__()
        self.doc = "LSTM_local_global"
        self.pose_dim = pose_dim
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.dev = dev

        self.embedding_fn = nn.Sequential(nn.Linear(3, embedding_dim), nn.Tanh())
        self.encoder_g = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.decoder_g = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.hidden2g = nn.Sequential(nn.Linear(h_dim, 3))

    def forward(self, global_s=None, pred_len=14):

        seq_len, batch, l = global_s.shape
        state_tuple_g = (torch.zeros(self.num_layers, batch, self.h_dim, device=self.dev, dtype=torch.float64),
                       torch.zeros(self.num_layers, batch, self.h_dim, device=self.dev, dtype=torch.float64))

        global_s = global_s.contiguous()
        output_g, state_tuple_g = self.encoder_g(self.embedding_fn(global_s.view(-1,3)).view(seq_len, batch, self.embedding_dim), state_tuple_g)

        pred_s_g = torch.tensor([], device=self.dev)
        last_s_g = global_s[-1].unsqueeze(0) 
        for _ in range(pred_len):

             output_g, state_tuple_g = self.decoder_g(self.embedding_fn(last_s_g.view(-1,3)).view(1, batch, self.embedding_dim), state_tuple_g)
             curr_s_g = self.hidden2g(output_g.view(-1, self.h_dim))
             pred_s_g = torch.cat((pred_s_g, curr_s_g.unsqueeze(0)), dim=0)
             last_s_g = curr_s_g.unsqueeze(0)

        return pred_s_g

class Encoder(nn.Module):
    def __init__(self, pose_dim=39, h_dim=32, latent_dim=16, num_layers=2, dropout=0.3, dev='cpu'):
        super(Encoder, self).__init__()

        self.pose_dim = pose_dim
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dev = dev

        self.encoder = nn.LSTM(pose_dim, h_dim, num_layers, dropout=dropout)
        self.FC_mean = nn.Linear(h_dim, latent_dim)
        self.FC_var = nn.Linear(h_dim, latent_dim)

    def forward(self, obs_s=None):

        batch = obs_s.size(1)
        state_tuple = (torch.zeros(self.num_layers, batch, self.h_dim, device=self.dev, dtype=torch.float64),
                       torch.zeros(self.num_layers, batch, self.h_dim, device=self.dev, dtype=torch.float64))
        output, state_tuple = self.encoder(obs_s, state_tuple)
        out = output[-1]
        mean = self.FC_mean(out)
        log_var = self.FC_var(out)  
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, pose_dim=39, h_dim=32, latent_dim=16, num_layers=1, dropout=0.3, dev='cpu'):
        super(Decoder, self).__init__()
        self.pose_dim = pose_dim
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dev = dev

        self.decoder = nn.LSTM(pose_dim, h_dim, num_layers, dropout=dropout)
        self.FC = nn.Sequential(nn.Linear(latent_dim, h_dim))
        self.mlp = nn.Sequential(nn.Linear(h_dim, pose_dim))

    def forward(self, obs_s=None, latent=None, pred_len=14):

        batch = obs_s.size(1)
        decoder_c = torch.zeros(self.num_layers, batch, self.h_dim, device=self.dev, dtype=torch.float64)
        last_s = obs_s[-1].unsqueeze(0)
        decoder_h = self.FC(latent).unsqueeze(0)
        decoder_h = decoder_h.repeat(self.num_layers, 1, 1)
        state_tuple = (decoder_h, decoder_c)

        preds_s = torch.tensor([], device=self.dev)
        for _ in range(pred_len):
            output, state_tuple = self.decoder(last_s, state_tuple)
            curr_s = self.mlp(output.view(-1, self.h_dim))
            preds_s = torch.cat((preds_s, curr_s.unsqueeze(0)), dim=0)
            last_s = curr_s.unsqueeze(0)
            
        return preds_s


class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, obs_s=None):
        mean, log_var = self.Encoder(obs_s=obs_s)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        preds_s = self.Decoder(obs_s=obs_s, latent=z)

        return preds_s, mean, log_var

def vae_loss_function(x, x_hat, mean, log_var):
    # BCE_loss = nn.BCELoss()
    # reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    assert x_hat.shape == x.shape
    reconstruction_loss = torch.mean(torch.norm(x - x_hat, dim=len(x.shape) - 1))
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + 0.01*KLD 

