
from torch import Tensor
import torch.nn.functional as f
import torch
from torch import nn
import numpy as np

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = (1./10.)*angle_rads[np.newaxis, ...]
    
    return torch.Tensor(pos_encoding).to(device='cuda')

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
    '''
    querry => ( ... ,seq_len_q,d_model) 
    key => ( ... ,seq_len_k,d_model)
    value => ( ... , seq_len_v, d_model)
    mask => (..., seq_len_q, seq_len_k)
    attn_weights => ( ... , seq_len_q, seq_len_k )
    out = attn_weights*values => (... , seq_len_q, d_model)
    '''
    
    seq_len_k = key.shape[-2]
    seq_len_v = value.shape[-2]
    
    assert(seq_len_k == seq_len_v)
    scale = query.size(-1) ** 0.5
    
    attn_weights = torch.matmul(query, key.transpose(-1,-2))
    attn_weights /= scale
    
    if mask is not None:
        attn_weights += mask*-1e9
    attn_weights = f.softmax(attn_weights, dim=-1) 
    out = torch.matmul(attn_weights, value)
    return  out, attn_weights



def create_look_ahead_mask(size):
    return torch.Tensor(np.triu( np.ones((size,size)),k=1))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
    
        assert d_model % self.num_heads == 0
    
        self.depth = d_model // self.num_heads
    
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
    
        self.linear = nn.Linear(self.num_heads*self.depth, d_model)
    
    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]
    
        q = self.q(q)  # (batch_size, seq_len, d_model)
        k = self.k(k)  # (batch_size, seq_len, d_model)
        v = self.v(v)  # (batch_size, seq_len, d_model)
    
        ########
        ###seq_len_q = q.shape[1]
        ###seq_len_k = k.shape[1]
        ###seq_len_v = v.shape[1]
        ###q = q.reshape(batch_size, seq_len_q, self.depth, self.num_heads).permute(0,3,1,2)
        ###k = k.reshape(batch_size, seq_len_k, self.depth, self.num_heads).permute(0,3,1,2)
        ###v = v.reshape(batch_size, seq_len_v, self.depth, self.num_heads).permute(0,3,1,2)
        ########
        q = q.reshape(batch_size, -1, self.num_heads, self.depth).permute(0,2,1,3) #(batch_size,num_heads,seq_len_q,depth)
        k = k.reshape(batch_size, -1, self.num_heads, self.depth).permute(0,2,1,3) #(batch_size,num_heads,seq_len_k,depth)
        v = v.reshape(batch_size, -1, self.num_heads, self.depth).permute(0,2,1,3) #(batch_size,num_heads,seq_len_v,depth)
    
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    
        scaled_attention = scaled_attention.permute(0,2,1,3) # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model) #(batch_size, seq_len_q, d_model)

        output = self.linear(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return nn.Sequential(
        nn.Linear(d_model, dff),  # (batch_size, seq_len, dff)
        nn.ReLU(),
        nn.Linear(dff, d_model)  # (batch_size, seq_len, d_model)
  )


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
    
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
    
    def forward(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        if(training): attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        if(training): ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
        return out2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
 
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
    
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)
    
    
    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        if(training): attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)
    
        attn2, attn_weights_block2 = self.mha2(
           enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        if(training): attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        if(training): ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
        return out3, attn_weights_block1, attn_weights_block2



class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
    
        #self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)
    
    
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)])
        #self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
        #               for _ in range(num_layers)]
  
        self.dropout = nn.Dropout(rate)
        
    def forward(self, x, training, mask):

        seq_len = x.shape[1]
    
        # adding embedding and position encoding.
        #x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[:, :seq_len, :]

        if(training): x = self.dropout(x)
        
        for layer in self.enc_layers:
            #print("layer {}".format(layer))
            x = layer(x, training, mask)
        #for i in range(self.num_layers):
        #    x = self.enc_layers[i](x, training, mask)
    
        return x  # (batch_size, input_seq_len, d_model)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff,
               maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
    
        #self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)
    
    def forward(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

        seq_len = x.shape[1]
        attention_weights = {}
    
        #x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[:, :seq_len, :]
    
        if(training): x = self.dropout(x)
        for i,layer in enumerate(self.dec_layers):
            x, block1, block2 = layer(x, enc_output, training,
                                             look_ahead_mask, padding_mask)
      
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        #for i in range(self.num_layers):
            #x, block1, block2 = self.dec_layers[i](x, enc_output, training,
            #                                 look_ahead_mask, padding_mask)
      
            #attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            #attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, 
               pe_input=1000, pe_target=1000, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           pe_target, rate)

        self.final_layer = nn.Linear(d_model, d_model)
        self.activation = nn.Hardtanh(min_val=-100, max_val=100)
        self.vlinear = nn.Linear(d_model, d_model//4)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
        #v = self.vlinear(dec_output)
        v = self.vlinear(dec_output.detach())
        v = self.sigmoid(v)
        #final_output = self.activation(final_output)
        return dec_output, v, attention_weights


