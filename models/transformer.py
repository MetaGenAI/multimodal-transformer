import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, device=None):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        self.lpe = nn.Embedding(max_len+1, d_model)
        self.indices = torch.arange(max_len).unsqueeze(1) + 1
        if device is not None:
            self.indices = self.indices.to(self.device)

    def forward(self, x, indices = None):
        if indices is None:
            indices = self.indices[:x.size(0),:]
            indices = self.dropout(indices)
        x = x + self.lpe(indices)
        return self.dropout(x)

class TransformerCausalModel(nn.Module):

    def __init__(self, dout, dinp, nhead, dhid, nlayers, dropout=0.5,device=None):
        super(TransformerCausalModel, self).__init__()
        self.device = device
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(dinp, dropout, device=self.device)
        self.encoder1 = nn.Linear(dinp, dhid)
        encoder_layers = TransformerEncoderLayer(dhid, nhead, dhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, dinp)
        self.dinp = dinp
        self.decoder = nn.Linear(dhid, dout)

        self.init_weights()

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
        # src = self.encoder(src) * math.sqrt(self.dinp)
        src *= math.sqrt(self.dinp)
        src = self.pos_encoder(src)
        # src = self.encoder1(src)
        # print(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
