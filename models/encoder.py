import torch
import torch.nn as nn

'''
encoder 整体是一个双向lstm，
输入就是格式：(batch_size, 序列长度，序列字符embedding 长度)
输出: 参数 q 和 logvar，大小为(batch_size, z_dim)，表示学习到的隐式空间。
'''

def build_encoder(enc_type, 
                  **E_args):
    if enc_type == 'gru':
        encoder = GRUEncoder(**E_args)
    else:
        raise ValueError('Please use GRU Encoder')
    return encoder

class GRUEncoder(nn.Module):
    """
    Encoder is GRU with FC layers connected to last hidden unit
    """
    def __init__(self, 
                 emb_dim,
                 h_dim,
                 z_dim,
                 biGRU,
                 layers,
                 p_dropout):
        super(GRUEncoder, self).__init__()
        self.rnn = nn.GRU(input_size=emb_dim, 
                          hidden_size=h_dim, 
                          num_layers=layers,
                          dropout=p_dropout,
                          bidirectional=biGRU,
                          batch_first=True)
        # Bidirectional GRU has 2*hidden_state
        self.biGRU_factor = 2 if biGRU else 1
        self.biGRU = biGRU
        # Reparametrization
        self.q_mu = nn.Linear(self.biGRU_factor*h_dim, z_dim)
        self.q_logvar = nn.Linear(self.biGRU_factor*h_dim, z_dim)

    def forward(self, x):
        """
        Inputs is embeddings of: mbsize x seq_len x emb_dim
        """
        _, h = self.rnn(x, None)
        if self.biGRU:
            # Concatenates features from Forward and Backward
            # Uses the highest layer representation
            h = torch.cat((h[-2,:,:], 
                           h[-1,:,:]), 1)
        # Forward to latent
        h = h.view(-1, h.shape[-1])
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)
        return mu, logvar