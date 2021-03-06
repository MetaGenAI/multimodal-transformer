import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
import numpy as np

#class PositionalEncoding(nn.Module):
#
#    def __init__(self, d_model, dropout=0.1, max_len=5000, device=None):
#        super(PositionalEncoding, self).__init__()
#        self.device = device
#        self.dropout = nn.Dropout(p=dropout)
#        self.lpe = nn.Embedding(max_len+1, d_model)
#        # self.weight = None
#        self.indices = torch.arange(max_len).unsqueeze(1) + 1
#        if device is not None:
#            self.indices = self.indices.to(self.device)
#
#    def init_weights(self):
#        initrange = 0.1
#        self.lpe.weight.data.uniform_(-initrange, initrange)
#
#    def forward(self, x, indices = None):
#        np.save(str(uuid.uuid4())+".np",self.lpe.weight.data.cpu().numpy())
#        if indices is None:
#            indices = self.indices[:x.size(0),:]
#            indices = self.dropout(indices)
#        x = x + self.lpe(indices)
#        return self.dropout(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, device=None):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term1 = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term2 = torch.exp(torch.arange(0, (d_model//2)*2, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term1)
        pe[:, 1::2] = torch.cos(position * div_term2)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerCausalModel(nn.Module):

    def __init__(self, dout, dinp, nhead, dhid, nlayers, dropout=0.5,device=None,use_pos_emb=False,input_length=0):
        super(TransformerCausalModel, self).__init__()
        self.device = device
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.encoder1 = nn.Linear(dinp, dhid)
        #self.pos_encoder = PositionalEncoding(dhid, dropout, device=self.device)
        #encoder_layers = TransformerEncoderLayer(dhid, nhead, 2048, dropout)
        encoder_layers = TransformerEncoderLayer(dhid, nhead, dhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, dinp)
        self.dinp = dinp
        self.dhid = dhid
        self.decoder = nn.Linear(dhid, dout)
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            assert input_length > 0
            self.pos_emb = nn.Parameter((torch.zeros(input_length,input_length)/np.sqrt(dinp)).to(self.device))

        self.init_weights()
        #self.pos_encoder.init_weights()

    def generate_square_subsequent_mask(self, sz, prefix_length = 1):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask[:,:prefix_length] = 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder1(src)
        #src *= math.sqrt(self.dhid)
        #src = self.pos_encoder(src)
        #src /= math.sqrt(self.dhid)
        # print(src)
        # print(torch.mm(src[:,0,:],src[:,0,:].T))
        if self.use_pos_emb:
            #print(self.pos_emb)
            src_mask += self.pos_emb
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
