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

        self.input_mods = input_mods = self.opt.input_modalities.split(",")
        self.output_mods = output_mods = self.opt.output_modalities.split(",")
        self.dins = dins = [int(x) for x in self.opt.dins.split(",")]
        self.douts = douts = [int(x) for x in self.opt.douts.split(",")]
        self.input_lengths = input_lengths = [int(x) for x in self.opt.input_lengths.split(",")]
        self.predicted_inputs = predicted_inputs = [int(x) for x in self.opt.predicted_inputs.split(",")]
        self.output_lengths = output_lengths = [int(x) for x in self.opt.output_lengths.split(",")]
        self.output_time_offsets = output_time_offsets = [int(x) for x in self.opt.output_time_offsets.split(",")]
        self.input_time_offsets = input_time_offsets = [int(x) for x in self.opt.input_time_offsets.split(",")]
        if len(output_time_offsets) < len(output_mods):
            if len(output_time_offsets) == 1:
                output_time_offsets = output_time_offsets*len(output_mods)
            else:
                raise Exception("number of output_time_offsets doesnt match number of output_mods")

        if len(input_time_offsets) < len(input_mods):
            if len(input_time_offsets) == 1:
                input_time_offsets = input_time_offsets*len(input_mods)
            else:
                raise Exception("number of input_time_offsets doesnt match number of input_mods")

        if len(predicted_inputs) < len(input_mods):
            if len(predicted_inputs) == 1:
                predicted_inputs = predicted_inputs*len(input_mods)
            else:
                raise Exception("number of predicted_inputs doesnt match number of input_mods")

        self.input_mod_nets = []
        self.output_mod_nets = []
        self.module_names = []
        for i, mod in enumerate(input_mods):
            net = TransformerCausalModel(opt.dhid, dins[i], opt.nhead, opt.dhid, opt.nlayers, opt.dropout)
            name = "_input_"+mod
            setattr(self,"net"+name, net)
            self.input_mod_nets.append(net)
            self.module_names.append(name)
        for i, mod in enumerate(output_mods):
            net = TransformerCausalModel(douts[i], opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout)
            name = "_output_"+mod
            setattr(self,"net"+name, net)
            self.output_mod_nets.append(net)
            self.module_names.append(name)

        self.src_masks = []
        for i, mod in enumerate(input_mods):
            self.src_masks.append(self.input_mod_nets[i].generate_square_subsequent_mask(input_lengths[i], input_lengths[i]-predicted_inputs[i]).to(self.device))

        input_present_matrix = torch.zeros(len(input_mods),max(np.array(input_lengths)+np.array(input_time_offsets)))
        for i, mod in enumerate(input_mods):
            input_present_matrix[i,input_time_offsets[i]:input_time_offsets[i]+input_lengths[i]] = 1

        input_index_matrix = torch.cumsum(input_present_matrix.T.flatten(),0).reshape(-1,len(input_mods)).T

        input_indices = torch.tensor([])
        for i, mod in enumerate(input_mods):
            mod_indices = input_index_matrix[i,input_time_offsets[i]:input_time_offsets[i]+input_lengths[i]]
            input_indices = torch.cat([input_indices,mod_indices])

        input_indices = input_indices - 1
        input_indices = input_indices.long()


        self.output_mask = self.output_mod_nets[0].generate_square_subsequent_mask(sum(input_lengths), 0)
        self.output_mask = self.output_mask[input_indices.unsqueeze(0).T,input_indices.unsqueeze(0)]
        print(self.output_mask.shape)
        j=0
        for i, mod in enumerate(input_mods):
            self.output_mask[j:j+input_lengths[i]-predicted_inputs[i],:] = 1
            j+=input_lengths[i]

        self.output_mask = self.output_mask.to(self.device)

        self.criterion = nn.MSELoss()

        self.optimizers = [torch.optim.Adam([
            {'params': sum([[param for name, param in net.named_parameters() if name[-4:] == 'bias'] for net in self.input_mod_nets+self.output_mod_nets],[]),
             'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
            {'params': sum([[param for name, param in net.named_parameters() if name[-4:] != 'bias'] for net in self.input_mod_nets+self.output_mod_nets],[]),
             'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
        ])]
        self.loss_mse = None

    def name(self):
        return "Transformer"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--dhid', type=int, default=512)
        parser.add_argument('--dins', default=None)
        parser.add_argument('--douts', default=None)
        parser.add_argument('--predicted_inputs', default="0")
        parser.add_argument('--nlayers', type=int, default=6)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        return parser

    def set_input(self, data):
        # move multiple samples of the same song to the second dimension and the reshape to batch dimension
        # input_ = data['input']
        # target_ = data['target']
        self.input = []
        self.target = []
        for i,mod in enumerate(self.input_mods):
            input_ = data["in_"+mod]
            input_shape = input_.shape
            # It's coming as 0 batch dimension, 1 window dimension, 2 input channel dimension, 3 time dimension
            input_ = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3])).permute(2,0,1).to(self.device)
            self.input.append(input_)
        for i,mod in enumerate(self.output_mods):
            target_ = data["out_"+mod]
            target_shape = target_.shape
            target_ = target_.reshape((target_shape[0]*target_shape[1], target_shape[2], target_shape[3])).permute(2,0,1).to(self.device)
            self.target.append(target_)
        # feature_sizes = data['feature_sizes']

    def forward(self):
        latents = []
        input_begin_indices = {}
        j=0
        for i, mod in enumerate(self.input_mods):
            latents.append(self.input_mod_nets[i].forward(self.input[i],self.src_masks[i]))
            input_begin_indices[mod] = j
            j+=self.input_lengths[i]

        latent = torch.cat(latents)
        self.loss_mse = 0
        for i, mod in enumerate(self.output_mods):
            inp_index = self.input_mods.index(mod)
            j = input_begin_indices[mod]
            output_begin_index = j+self.input_lengths[inp_index]-self.predicted_inputs[inp_index]
            output = self.output_mod_nets[i].forward(latent,self.output_mask)[output_begin_index:output_begin_index+self.output_lengths[i]]
            self.loss_mse += self.criterion(output,self.target[i])

        self.metric_mse = self.loss_mse

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

        # 0 batch dimension, 1 input channel dimension, 2 time dimension
        # -> 0 time dimension, 1 batch dimenison, 2 channel dimension
        x = x.unsqueeze(0).permute(2,0,1).to(self.device)

        input_seq = x[:opt.input_seq_len].clone()
        # print(input_seq.shape)

        output_seq = None
        #self.eval()
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

        with torch.no_grad():
            for t in range(sequence_length-input_length-1):
            # for t in range(sequence_length):
                # time.sleep(1)
                print(t)
                next_prediction = self.net.forward(input_seq.float(),self.src_mask)[self.opt.prefix_length:self.opt.prefix_length+1].detach()
                #indices = np.random.choice(range(0,sequence_length-max(input_length,time_offset+output_length)),size=10,replace=True)
                #input_windows = [x[i:i+input_length] for i in indices]
                #next_prediction = self.net.forward(torch.cat([input_seq]+input_windows,1).float(),self.src_mask)[self.opt.prefix_length:self.opt.prefix_length+1,:1].detach()

                #next_prediction = self.net.forward(input_seq.float(),self.src_mask).detach()
                #input_temp=x.clone()
                #next_prediction = self.net.forward(input_temp[t:t+opt.input_seq_len].float(),self.src_mask).detach()
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
                            #print(j)
                            #input_seq[:,:,i:i+dmod] = torch.cat([input_seq[1:,:,i:i+dmod],next_prediction[:,:,j:j+dmod]],0)
                            input_seq[:,:,i:i+dmod] = torch.cat([input_seq[1:,:,i:i+dmod],x[opt.input_seq_len+t+1:opt.input_seq_len+t+2,:,i:i+dmod]],0)
                            #print(torch.mean((x[t+opt.prefix_length+1:t+opt.prefix_length+output_length+1,:,i:i+dmod]-next_prediction[self.opt.prefix_length:,:,j:j+dmod])**2))
                            #print(torch.mean((x[t+opt.prefix_length+1:t+opt.prefix_length+output_length+1,:,i:i+dmod]-next_prediction[self.opt.prefix_length:,:,j:j+dmod])**2))
                            print(torch.mean((x[t+opt.prefix_length+1:t+opt.prefix_length+1+1,:,i:i+dmod]-next_prediction[:,:,j:j+dmod])**2))
                            #print(x[0,0,0])
                            #print(output_length)
                            #print(x[t+opt.prefix_length+1:t+opt.prefix_length+output_length+1,:,i:i+dmod])
                            #print(next_prediction[self.opt.prefix_length:,:,j:j+dmod])
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
        for net in self.input_mod_nets:
            self.set_requires_grad(net, requires_grad=True)
        for net in self.output_mod_nets:
            self.set_requires_grad(net, requires_grad=True)
        self.forward()
        self.backward()
