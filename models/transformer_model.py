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
                self.output_time_offsets = output_time_offsets = output_time_offsets*len(output_mods)
            else:
                raise Exception("number of output_time_offsets doesnt match number of output_mods")

        if len(input_time_offsets) < len(input_mods):
            if len(input_time_offsets) == 1:
                self.input_time_offsets = input_time_offsets = input_time_offsets*len(input_mods)
            else:
                raise Exception("number of input_time_offsets doesnt match number of input_mods")

        if len(predicted_inputs) < len(input_mods):
            if len(predicted_inputs) == 1:
                self.predicted_inputs = predicted_inputs = predicted_inputs*len(input_mods)
            else:
                raise Exception("number of predicted_inputs doesnt match number of input_mods")

        self.input_mod_nets = []
        self.output_mod_nets = []
        self.module_names = []
        for i, mod in enumerate(input_mods):
            #net = TransformerCausalModel(opt.dhid, dins[i], opt.nhead, opt.dhid, opt.nlayers, opt.dropout)
            net = TransformerCausalModel(opt.dhid, dins[i], opt.nhead, opt.dhid, 2, opt.dropout, self.device).to(self.device)
            name = "_input_"+mod
            setattr(self,"net"+name, net)
            self.input_mod_nets.append(net)
            self.module_names.append(name)
        for i, mod in enumerate(output_mods):
            net = TransformerCausalModel(douts[i], opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout, self.device).to(self.device)
            name = "_output_"+mod
            setattr(self,"net"+name, net)
            self.output_mod_nets.append(net)
            self.module_names.append(name)

        #self.generate_masks()
        self.generate_full_masks()

        self.criterion = nn.MSELoss()

        print(opt.learning_rate)
        self.optimizers = [torch.optim.Adam([
            {'params': sum([[param for name, param in net.named_parameters() if name[-4:] == 'bias'] for net in self.input_mod_nets+self.output_mod_nets],[]),
             'lr': 1 * opt.learning_rate },  # bias parameters change quicker - no weight decay is applied
            {'params': sum([[param for name, param in net.named_parameters() if name[-4:] != 'bias'] for net in self.input_mod_nets+self.output_mod_nets],[]),
             'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
        ])]
        self.loss_mse = None

    def generate_full_masks(self):
        input_mods = self.input_mods
        output_mods = self.output_mods
        dins = self.dins
        douts = self.douts
        input_lengths = self.input_lengths
        predicted_inputs = self.predicted_inputs
        output_lengths = self.output_lengths
        output_time_offsets = self.output_time_offsets
        input_time_offsets = self.input_time_offsets
        self.src_masks = []
        src_pos_embs = []
        for i, mod in enumerate(input_mods):
            mask = torch.zeros(input_lengths[i],input_lengths[i]).to(self.device)
            pos_emb = nn.Parameter(torch.randn(*mask.shape).to(self.device))
            src_pos_embs.append(pos_emb)
            self.src_masks.append(mask+pos_emb)

        self.output_masks = []
        for i, mod in enumerate(output_mods):
            mask = torch.zeros(sum(input_lengths),sum(input_lengths)).to(self.device)
            self.output_masks.append(mask.to(self.device))

        self.src_pos_embs_params = nn.ParameterList(src_pos_embs)

    def generate_masks(self):
        input_mods = self.input_mods
        output_mods = self.output_mods
        dins = self.dins
        douts = self.douts
        input_lengths = self.input_lengths
        predicted_inputs = self.predicted_inputs
        output_lengths = self.output_lengths
        output_time_offsets = self.output_time_offsets
        input_time_offsets = self.input_time_offsets
        self.src_masks = []
        src_pos_embs = []
        for i, mod in enumerate(input_mods):
            mask = self.input_mod_nets[i].generate_square_subsequent_mask(input_lengths[i], input_lengths[i]-predicted_inputs[i]).to(self.device)
            pos_emb = nn.Parameter(torch.randn(*mask.shape).to(self.device))
            src_pos_embs.append(pos_emb)
            self.src_masks.append(mask+pos_emb)

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

        self.output_masks = []
        #out_pos_embs = []
        for i, mod in enumerate(output_mods):
            net = self.output_mod_nets[i]
            mask = net.generate_square_subsequent_mask(sum(input_lengths), 0)
            mask = mask[input_indices.unsqueeze(0).T,input_indices.unsqueeze(0)]
            j=0
            for i, mod in enumerate(input_mods):
                mask[j:j+input_lengths[i]-predicted_inputs[i],:] = float('-inf')
                j+=input_lengths[i]
            j=0
            for i, mod in enumerate(input_mods):
                mask[:,j:j+input_lengths[i]-predicted_inputs[i]] = 0
                j+=input_lengths[i]
            self.output_masks.append(mask.to(self.device))
        #for i, mod in enumerate(output_mods):
        #    mask = self.output_masks[i]
        #    pos_emb = nn.Parameter(torch.randn(*mask.shape).to(self.device))
        #    out_pos_embs.append(pos_emb)
        #    self.output_masks[i] += pos_emb

        self.src_pos_embs_params = nn.ParameterList(src_pos_embs)
        #self.out_pos_embs_params = nn.ParameterList(out_pos_embs)
        #self.pos_embs = self.pos_embs.to(self.device)


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
        self.inputs = []
        self.targets = []
        for i,mod in enumerate(self.input_mods):
            input_ = data["in_"+mod]
            input_shape = input_.shape
            # It's coming as 0 batch dimension, 1 window dimension, 2 input channel dimension, 3 time dimension
            input_ = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3])).permute(2,0,1).to(self.device)
            self.inputs.append(input_)
        for i,mod in enumerate(self.output_mods):
            target_ = data["out_"+mod]
            target_shape = target_.shape
            target_ = target_.reshape((target_shape[0]*target_shape[1], target_shape[2], target_shape[3])).permute(2,0,1).to(self.device)
            self.targets.append(target_)
        # feature_sizes = data['feature_sizes']

    def forward(self, calc_loss=True):
        latents = []
        input_begin_indices = {}
        j=0
        for i, mod in enumerate(self.input_mods):
            latents.append(self.input_mod_nets[i].forward(self.inputs[i],self.src_masks[i]))
            input_begin_indices[mod] = j
            j+=self.input_lengths[i]

        latent = torch.cat(latents)
        # print(latent.std())
        self.loss_mse = 0
        self.outputs = []
        for i, mod in enumerate(self.output_mods):
            inp_index = self.input_mods.index(mod)
            j = input_begin_indices[mod]
            output_begin_index = j+self.input_lengths[inp_index]-self.predicted_inputs[inp_index]
            #output = self.output_mod_nets[i].forward(latent,self.output_masks[i])[output_begin_index:output_begin_index+self.output_lengths[i]]
            output = self.output_mod_nets[i].forward(latent,self.output_masks[i])[:self.output_lengths[i]]
            self.outputs.append(output)
            if calc_loss:
                self.loss_mse += self.criterion(output,self.targets[i])

        self.metric_mse = self.loss_mse

    def generate(self, features):
        opt = self.opt

        inputs_ = []
        for i,mod in enumerate(self.input_mods):
            input_ = features["in_"+mod]
            #print(input_)
            input_ = torch.from_numpy(input_).float()
            input_shape = input_.shape
            # It's coming as 0 batch dimension, 1 window dimension, 2 input channel dimension, 3 time dimension
            input_ = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3])).permute(2,0,1).to(self.device)
            inputs_.append(input_)
            #if self.predicted_inputs[i]>0:
            #    self.input_lengths[i] = self.input_lengths[i] - self.predicted_inputs[i] + 1
            #    self.predicted_inputs[i] = 1

        #self.generate_masks()

        self.inputs = []
        input_tmp = []
        for i,mod in enumerate(self.input_mods):
            input_tmp.append(inputs_[i].clone()[self.input_time_offsets[i]:self.input_time_offsets[i]+self.input_lengths[i]])

        self.eval()
        output_seq = []
        # sequence_length = inputs_[0].shape[0]
        sequence_length = inputs_[1].shape[0]
        with torch.no_grad():
            for t in range(min(512,sequence_length-max(self.input_lengths)-1)):
            # for t in range(sequence_length-max(self.input_lengths)-1):
            # for t in range(sequence_length):
                print(t)
                self.inputs = [x.clone() for x in input_tmp]
                #for i,mod in enumerate(self.input_mods):
                #    print(self.inputs[i])
                self.forward(False)
                if t == 0:
                    for i,mod in enumerate(self.output_mods):
                        output = self.outputs[i]
                        # output[:,0,:-3] = torch.clamp(output[:,0,:-3],-3,3)
                        output_seq.append(output[:1].detach().clone())
                        # output_seq.append(inputs_[i][t+self.input_time_offsets[i]+self.input_lengths[i]:t+self.input_time_offsets[i]+self.input_lengths[i]+1]+0.15*torch.randn(1,219).cuda())
                else:
                    for i,mod in enumerate(self.output_mods):
                        # output_seq[i] = torch.cat([output_seq[i], inputs_[i][t+self.input_time_offsets[i]+self.input_lengths[i]:t+self.input_time_offsets[i]+self.input_lengths[i]+1]+0.15*torch.randn(1,219).cuda()])
                        output = self.outputs[i]
                        output_seq[i] = torch.cat([output_seq[i], output[:1].detach().clone()])

                        # output[:,0,:-3] = torch.clamp(output[:,0,:-3],-3,3)
                        # print(self.outputs[i][:1])
                        #print(inputs_[i][t+self.input_time_offsets[i]+self.input_lengths[i]:t+self.input_time_offsets[i]+self.input_lengths[i]+1]+0.15*torch.randn(1,219).cuda())
                    #output_seq = torch.cat([output_seq, x[opt.input_seq_len+t+1:opt.input_seq_len+t+2,:,:219]])
                if t < sequence_length-1:
                    for i,mod in enumerate(self.input_mods):
                        if self.predicted_inputs[i] > 0:
                            j = self.output_mods.index(mod)
                            #input_tmp[i] = torch.cat([input_tmp[i][1:],self.outputs[j][-1:].detach().clone()],0)
                            output = self.outputs[i]
                            # print(output.shape)
                            # output[:,0,:-3] = torch.clamp(output[:,0,:-3],-3,3)
                            input_tmp[i] = torch.cat([input_tmp[i][1:-self.predicted_inputs[i]+1],output.detach().clone()],0)
                            # input_tmp[i] = torch.cat([input_tmp[i][1:],inputs_[i][t+self.input_time_offsets[i]+self.input_lengths[i]:t+self.input_time_offsets[i]+self.input_lengths[i]+1]],0)
                            print(torch.mean((inputs_[i][t+self.input_time_offsets[i]+self.input_lengths[i]-self.predicted_inputs[i]+1:t+self.input_time_offsets[i]+self.input_lengths[i]-self.predicted_inputs[i]+1+1]-self.outputs[j][:1].detach().clone())**2))
                            #print(torch.mean((inputs_[i][t+self.input_time_offsets[i]+self.input_lengths[i]-self.predicted_inputs[i]+1:t+self.input_time_offsets[i]+self.input_lengths[i]-self.predicted_inputs[i]+1+self.output_lengths[j]]-self.outputs[j].detach().clone())**2))
                            # input_seq[:,:,i:i+dmod] = torch.cat([input_seq[1:,:,i:i+dmod],x[opt.input_seq_len+t+1:opt.input_seq_len+t+2,:,i:i+dmod]],0)
                            #print(input_tmp[i][t+self.input_time_offsets[i]+self.input_lengths[i]+1:t+self.input_time_offsets[i]+self.input_lengths[i]+1+1])
                            #print(input_tmp[i].shape)
                        else:
                            input_tmp[i] = torch.cat([input_tmp[i][1:],inputs_[i][self.input_time_offsets[i]+self.input_lengths[i]+t:self.input_time_offsets[i]+self.input_lengths[i]+t+1]],0)

                # torch.cuda.empty_cache()

            #output_seq = x[:,:,:219]

            return output_seq


    def backward(self):
        self.optimizers[0].zero_grad()
        self.loss_mse.backward()
        for net in self.input_mod_nets:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        for net in self.output_mod_nets:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        self.optimizers[0].step()

    def optimize_parameters(self):
        for net in self.input_mod_nets:
            self.set_requires_grad(net, requires_grad=True)
        for net in self.output_mod_nets:
            self.set_requires_grad(net, requires_grad=True)
        self.forward()
        self.backward()
