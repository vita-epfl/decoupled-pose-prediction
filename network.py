import torch.nn as nn
import torch
 

#LSTM model for posetrack data set
class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        
        #2 layers of LSTM cell for poses 
        self.lstm_pose_layer_1 = nn.LSTMCell(input_size=28, hidden_size=args.hidden_size)
        self.lstm_pose_layer_2 = nn.LSTMCell(input_size=args.hidden_size, hidden_size=args.hidden_size)

        #2 layers of LSTM cell for speed 
        self.lstm_vel_layer_1 = nn.LSTMCell(input_size=28, hidden_size=args.hidden_size)
        self.lstm_vel_layer_2 = nn.LSTMCell(input_size=args.hidden_size, hidden_size=args.hidden_size)
        
        #
        self.fc_vel = nn.Linear(in_features=args.hidden_size, out_features=28)
        
        self.hardtanh = nn.Hardtanh(min_val=-1*args.hardtanh_limit,max_val=args.hardtanh_limit)
        self.relu = nn.LeakyReLU() 
        self.softmax = nn.Softmax(dim=1)
        
        self.mask_encoder = nn.LSTM(input_size=14, hidden_size=args.hidden_size)
        self.mask_decoder = nn.LSTMCell(input_size=14, hidden_size=args.hidden_size)
        self.fc_mask    = nn.Linear(in_features=args.hidden_size, out_features=14)
        
        self.sigmoid = nn.Sigmoid()
        self.args = args
        
    def forward(self, pose=None, vel=None, mask=None):


        n_batch, seq_in, l = pose.shape

        #initializing hidden and cell states
        hidden_vel_layer_1 = torch.zeros(n_batch, self.args.hidden_size, device="cuda")
        cell_vel_layer_1 = torch.zeros(n_batch, self.args.hidden_size, device="cuda")
        hidden_vel_layer_2 = torch.zeros(n_batch, self.args.hidden_size, device="cuda")
        cell_vel_layer_2 = torch.zeros(n_batch, self.args.hidden_size, device="cuda")

        hidden_pose_layer_1 = torch.zeros(n_batch, self.args.hidden_size, device="cuda")
        cell_pose_layer_1 = torch.zeros(n_batch, self.args.hidden_size, device="cuda")
        hidden_pose_layer_2 = torch.zeros(n_batch, self.args.hidden_size, device="cuda")
        cell_pose_layer_2 = torch.zeros(n_batch, self.args.hidden_size, device="cuda")

        vel = vel.permute(1,0,2)
        pose = pose.permute(1,0,2)
        for i in range(seq_in-1):
             hidden_pose_layer_1, cell_pose_layer_1 = self.lstm_pose_layer_1(pose[i+1], (hidden_pose_layer_1, cell_pose_layer_1) )
             hidden_pose_layer_2, cell_pose_layer_2 = self.lstm_pose_layer_2(hidden_pose_layer_1, (hidden_pose_layer_2, cell_pose_layer_2) )
        
             hidden_vel_layer_1, cell_vel_layer_1 = self.lstm_vel_layer_1(vel[i], (hidden_vel_layer_1, cell_vel_layer_1) )
             hidden_vel_layer_2, cell_vel_layer_2 = self.lstm_vel_layer_2(hidden_vel_layer_1, (hidden_vel_layer_2, cell_vel_layer_2) )

        hidden_vel_layer_1 = hidden_vel_layer_1 + hidden_pose_layer_1
        hidden_vel_layer_2 = hidden_vel_layer_2 + hidden_pose_layer_2

        #encoding the observed masks
        _, (hidden_dec2, cell_dec2) = self.mask_encoder(mask.permute(1,0,2))
        hidden_dec2 = hidden_dec2.squeeze(0)
        cell_dec2 = cell_dec2.squeeze(0)
        outputs = []
        mask_outputs    = torch.tensor([], device=self.args.device)
        MaskDec_inp = mask[:,-1,:]
        
        x = vel[-1]
        vel_outputs = torch.tensor([], device='cuda')
        for i in range(14): 

             hidden_vel_layer_1, cell_vel_layer_1 = self.lstm_vel_layer_1(x, (hidden_vel_layer_1, cell_vel_layer_1) )
             hidden_vel_layer_2, cell_vel_layer_2 = self.lstm_vel_layer_2(hidden_vel_layer_1, (hidden_vel_layer_2, cell_vel_layer_2) )

             out = self.hardtanh(self.fc_vel(hidden_vel_layer_2))
             vel_outputs = torch.cat((vel_outputs, out.unsqueeze(1)), dim=1)
             x = out.detach()
             #mask
             hidden_dec2 = hidden_dec2 + hidden_vel_layer_2
             cell_dec2 = cell_dec2 + cell_vel_layer_2
             hidden_dec2, cell_dec2 = self.mask_decoder(MaskDec_inp, (hidden_dec2, cell_dec2))
             mask_output  = self.sigmoid(self.fc_mask(hidden_dec2))
             mask_outputs = torch.cat((mask_outputs, mask_output.unsqueeze(1)), dim = 1)
             MaskDec_inp  = mask_output #.detach()

        vel = vel.permute(1,0,2)
        pose = pose.permute(1,0,2)

            
        outputs.append(vel_outputs)    
        outputs.append(mask_outputs) 
        return tuple(outputs)



