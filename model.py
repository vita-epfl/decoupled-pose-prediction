import torch
import torch.nn as nn
import numpy as np

MAX_VAL = 190 
THRESHOLD = 50

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=28, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=False, dropout=0.0
        ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = curr_rel_pos #self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h



class LSTM(nn.Module):
    def __init__(self, embedding_dim=28, h_dim=64, num_layers=1, bottleneck_dim = 128, mlp_dim = 128, dropout=0.):

        super(LSTM, self).__init__()
         
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.pooling = True

        self.encoder_pose = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.decoder_pose = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.hidden2pose = nn.Sequential(nn.Linear(h_dim, embedding_dim), nn.Hardtanh(min_val=-MAX_VAL, max_val=MAX_VAL))
        #self.hidden2pose = nn.Sequential(nn.Linear(h_dim, embedding_dim), nn.Tanh())
        
        self.encoder_mask = nn.LSTM(embedding_dim//2, h_dim, num_layers, dropout=dropout)
        self.decoder_mask = nn.LSTM(embedding_dim//2, h_dim, num_layers, dropout=dropout)
        self.hidden2mask = nn.Sequential(nn.Linear(h_dim, embedding_dim//2), nn.Sigmoid())
        self.l1 = nn.Linear(2*h_dim, h_dim)
        

        if(self.pooling):
            self.pool_net = PoolHiddenNet(h_dim=h_dim, mlp_dim=mlp_dim, bottleneck_dim=bottleneck_dim)
            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                    mlp_dims,
                    activation='relu',
                    batch_norm=True,
                    dropout=0.
                )

        
    def forward(self, obs_pose=None, obs_speed=None, obs_mask=None, start_end_idx=None, pred_len=14):


        batch = obs_speed.size(1)
        state_tuple_p = (torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
                       torch.zeros(self.num_layers, batch, self.h_dim).cuda())
        state_tuple_m = (torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
                       torch.zeros(self.num_layers, batch, self.h_dim).cuda())

        output, state_tuple_p = self.encoder_pose(obs_speed, state_tuple_p)
        output, state_tuple_m = self.encoder_mask(obs_mask, state_tuple_m)


        pred_pose = torch.tensor([], device='cuda')
        pred_speed = torch.tensor([], device='cuda')
        pred_mask = torch.tensor([], device='cuda')

        curr_pose = obs_pose[-1]
        last_speed = obs_speed[-1].unsqueeze(0) 
        last_mask = obs_mask[-1].unsqueeze(0)
        for _ in range(pred_len):

             if(self.pooling):
                  decoder_h, decoder_c = state_tuple_p
                  pool_h = self.pool_net(decoder_h, start_end_idx, curr_pose)   
                  decoder_h = torch.cat([decoder_h.view(-1, self.h_dim), pool_h], dim=1)
                  decoder_h = self.mlp(decoder_h)
                  decoder_h = torch.unsqueeze(decoder_h, 0)
                  state_tuple_p = (decoder_h, decoder_c)
                  decoder_h_m, decoder_c_m = state_tuple_m
                  state_tuple_m = (self.l1(torch.cat((decoder_h_m, decoder_h), dim=-1)), decoder_c_m)
                  #state_tuple_m = (decoder_h_m + decoder_h, decoder_c_m)
                  #print("deb: ", decoder_h[0,0])


             output, state_tuple_p = self.decoder_pose(last_speed, state_tuple_p)
             #curr_speed = MAX_VAL*self.hidden2pose(output.view(-1, self.h_dim))
             curr_speed = self.hidden2pose(output.view(-1, self.h_dim))
             curr_pose = curr_pose + curr_speed
             pred_pose = torch.cat((pred_pose, curr_pose.unsqueeze(0).detach()), dim=0)
             pred_speed = torch.cat((pred_speed, curr_speed.unsqueeze(0)), dim=0)
             last_speed = curr_speed.unsqueeze(0)
 
             output, state_tuple_m = self.decoder_mask(last_mask, state_tuple_m)
             curr_mask = self.hidden2mask(output.view(-1, self.h_dim))
             pred_mask = torch.cat((pred_mask, curr_mask.unsqueeze(0)), dim=0)
             last_mask = curr_mask.unsqueeze(0)

        pred_pose = pred_pose.view(pred_len, batch, self.embedding_dim//2, 2)
        pred_mask[torch.sqrt(torch.sum(pred_pose**2, dim=-1)) < THRESHOLD] = 0.01
        pred_pose = pred_pose.view(pred_len, batch, self.embedding_dim)
        
        return pred_speed, pred_mask 


