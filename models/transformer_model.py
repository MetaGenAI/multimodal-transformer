import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from .base_model import BaseModel
# from .transformer.Models import Transformer
from .transformer import TransformerCausalModel
from models import constants
import time
# from transformer.Translator import Translator

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    # gold = gold.contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    # print(pred.shape, gold.shape)
    # print(pred, gold)
    # name = input("Enter your name: ")   # Python 3
    non_pad_mask = gold.ne(constants.PAD_STATE)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(constants.PAD_STATE)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=constants.PAD, reduction='mean')

    return loss




class TransformerModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.loss_names = ['mse']
        # self.metric_names = ['accuracy']
        self.metric_names = ['mse']
        self.module_names = ['']  # changed from 'model_names'
        self.schedulers = []
        # self.net = Transformer(
        #         d_tgt=opt.d_tgt,
        #         d_src=opt.d_src,
        #         n_tgt_vocab=opt.tgt_vocab_size,
        #         n_src_vocab=0,
        #         len_max_seq=opt.max_token_seq_len,
        #         src_vector_input=opt.src_vector_input,
        #         tgt_vector_input=opt.tgt_vector_input,
        #         tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        #         emb_src_tgt_weight_sharing=opt.embs_share_weight,
        #         d_k=opt.d_k,
        #         d_v=opt.d_v,
        #         d_model=opt.d_model,
        #         d_word_vec=opt.d_word_vec,
        #         d_inner=opt.d_inner_hid,
        #         n_layers=opt.n_layers,
        #         n_head=opt.n_head,
        #         dropout=opt.dropout)
        self.net = TransformerCausalModel(opt.dout, opt.din, opt.nhead, opt.dhid, opt.nlayers, opt.dropout)
        self.src_mask = self.net.generate_square_subsequent_mask(self.opt.input_seq_len, self.opt.prefix_length).to(self.device)
        self.criterion = nn.MSELoss()

        self.optimizers = [torch.optim.Adam([
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
             'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
        ])]
        self.loss_mse = None

    def name(self):
        return "Transformer"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--dhid', type=int, default=512)
        parser.add_argument('--din', type=int, default=100)
        parser.add_argument('--dout', type=int, default=100)
        parser.add_argument('--prefix_length', type=int, default=1)
        parser.add_argument('--proj_share_weight', action='store_true')
        parser.add_argument('--embs_share_weight', action='store_true')
        parser.add_argument('--label_smoothing', action='store_true')
        parser.add_argument('--d_k', type=int, default=64)
        parser.add_argument('--d_v', type=int, default=64)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--d_word_vec', type=int, default=512)
        # parser.add_argument('--d_inner_hid', type=int, default=2048)
        parser.add_argument('--nlayers', type=int, default=6)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        return parser

    def set_input(self, data):
        # move multiple samples of the same song to the second dimension and the reshape to batch dimension
        input_ = data['input']
        target_ = data['target']
        feature_sizes = data['feature_sizes']

        target_shape = target_.shape
        input_shape = input_.shape
        # 0 batch dimension, 1 window dimension, 2 input channel dimension, 3 time dimension
        # print(input_shape)
        self.input = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3])).permute(2,0,1).to(self.device)

        self.target = target_.reshape((target_shape[0]*target_shape[1], target_shape[2], target_shape[3])).permute(2,0,1).to(self.device)

    def forward(self):
        self.output = self.net.forward(self.input.float(),self.src_mask)
        self.metric_mse = self.loss_mse = self.criterion(self.output[self.opt.prefix_length:],self.target)

    def generate(self, features, temperature, mod_sizes = {}, predicted_mods = [], use_beam_search=False):
        opt = self.opt

        features = torch.from_numpy(features)


        input_length = self.opt.input_seq_len
        output_length = self.opt.output_seq_len
        time_offset = self.opt.output_time_offset
        input_mods = self.opt.input_modalities.split(",")
        output_mods = self.opt.output_modalities.split(",")

        input_features = None
        output_features = None

        x = features

        # we pad the song features with zeros to start generating from the beginning
        assert len(x.shape) == 2
        sequence_length = x.shape[1]
        # x = np.concatenate((np.zeros(( x.shape[0],max(0,time_offset) )),x),1)
        # # we also pad at the end to allow generation to be of the same length of sequence
        # x = np.concatenate((x,np.zeros(( x.shape[0],max(0,input_length-(time_offset)) ))),1)

        x = torch.tensor(x)

        # 0 batch dimension, 1 input channel dimension, 2 time dimension
        # -> 0 time dimension, 1 batch dimenison, 2 channel dimension
        x = x.unsqueeze(0).permute(2,0,1).to(self.device)

        input_seq = x[:opt.input_seq_len].clone()
        # print(input_seq.shape)

        output_seq = None
        self.eval()
        out_mod_indices = {}
        in_mod_indices = {}
        i=0
        for mod in input_mods:
            in_mod_indices[mod] = i
            dmod = mod_sizes[mod]
            i += dmod
        i=0
        for mod in output_mods:
            out_mod_indices[mod] = i
            dmod = mod_sizes[mod]
            i += dmod

        for t in range(sequence_length-input_length-1):
        # for t in range(sequence_length):
            # time.sleep(1)
            print(t)
            next_prediction = self.net.forward(input_seq.float(),self.src_mask)[self.opt.prefix_length:self.opt.prefix_length+1].detach()
            if t == 0:
                output_seq = next_prediction
            else:
                output_seq = torch.cat([output_seq, next_prediction])
                #output_seq = torch.cat([output_seq, x[opt.input_seq_len+t+1:opt.input_seq_len+t+2,:,:219]])
            if t < sequence_length-1:
                for mod in input_mods:
                    dmod = mod_sizes[mod]
                    i = in_mod_indices[mod]
                    if mod in predicted_mods:
                        j = out_mod_indices[mod]
                        print(j)
                        #input_seq[:,:,i:i+dmod] = torch.cat([input_seq[1:,:,i:i+dmod],next_prediction[:,:,j:j+dmod]],0)
                        input_seq[:,:,i:i+dmod] = torch.cat([input_seq[1:,:,i:i+dmod],x[opt.input_seq_len+t+1:opt.input_seq_len+t+2,:,i:i+dmod]],0)
                        print(torch.mean((x[opt.input_seq_len+t+1:opt.input_seq_len+t+2,:,i:i+dmod]-next_prediction[:,:,j:j+dmod])**2))
                    else:
                        input_seq[:,:,i:i+dmod] = torch.cat([input_seq[1:,:,i:i+dmod],x[opt.input_seq_len+t+1:opt.input_seq_len+t+2,:,i:i+dmod]],0)

            # torch.cuda.empty_cache()

        #output_seq = x[:,:,:219]

        return output_seq


    def backward(self):
        self.optimizers[0].zero_grad()
        self.loss_mse.backward()
        self.optimizers[0].step()

    def optimize_parameters(self):
        self.set_requires_grad(self.net, requires_grad=True)
        self.forward()
        self.backward()
