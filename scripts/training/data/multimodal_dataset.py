import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(os.path.join(THIS_DIR, os.pardir, os.pardir), os.pardir)))
# print(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

sys.path.append(ROOT_DIR)

from pathlib import Path
from itertools import tee
import numpy as np
import torch
from .base_dataset import BaseDataset
import json
from math import floor, ceil
import models.constants as constants


class MultimodalDataset(BaseDataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        data_path = Path(opt.data_dir)
        if not data_path.is_dir():
            raise ValueError('Invalid directory:'+opt.data_dir)

        temp_base_filenames = [x[:-1] for x in open(data_path.joinpath("base_filenames.txt"), "r").readlines()]
        self.base_filenames = []

        input_mods = self.opt.input_modalities.split(",")
        output_mods = self.opt.output_modalities.split(",")
        self.input_lengths = input_lengths = [int(x) for x in self.opt.input_lengths.split(",")]
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

        self.input_features = {input_mod:{} for input_mod in input_mods}
        self.output_features = {output_mod:{} for output_mod in output_mods}

        min_length = max(max(np.array(input_lengths) + np.array(input_time_offsets)), max(np.array(output_time_offsets) + np.array(output_lengths)) ) - min(0,min(output_time_offsets))
        print(min_length)

        fix_lengths = False

        #Get the list of files containing features (in numpy format for now), and populate the dictionaries of input and output features (separated by modality)
        for base_filename in temp_base_filenames:
            file_too_short = False
            for i, mod in enumerate(input_mods):
                feature_file = data_path.joinpath(base_filename+"."+mod+".npy")
                #print(feature_file)
                try:
                    features = np.load(feature_file)
                    length = features.shape[0]
                    #print(features.shape)
                    #print(length)
                    if not fix_lengths:
                        if i == 0:
                            length_0 = length
                        else:
                            assert length == length_0
                    if length < min_length:
                        # print("Smol sequence "+base_filename+"."+mod+"; ignoring..")
                        file_too_short = True
                        break
                except FileNotFoundError:
                    raise Exception("An unprocessed input feature found "+base_filename+"."+mod+"; need to run preprocessing script before starting to train with them")

            if file_too_short: continue

            for i, mod in enumerate(output_mods):
                feature_file = data_path.joinpath(base_filename+"."+mod+".npy")
                try:
                    features = np.load(feature_file)
                    length = features.shape[0]
                    if not fix_lengths:
                        if i == 0:
                            length_0 = length
                        else:
                            assert length == length_0
                    if length < min_length:
                        # print("Smol sequence "+base_filename+"."+mod+"; ignoring..")
                        file_too_short = True
                        break
                except FileNotFoundError:
                    raise Exception("An unprocessed output feature found "+base_filename+"."+mod+"; need to run preprocessing script before starting to train with them")

            if file_too_short: continue

            for mod in input_mods:
                feature_file = data_path.joinpath(base_filename+"."+mod+".npy")
                self.input_features[mod][base_filename] = feature_file

            if fix_lengths:
                shortest_length = 99999999999
                for mod in input_mods:
                    length = np.load(self.input_features[mod][base_filename]).shape[0]
                    if length < shortest_length:
                        shortest_length = length
                for mod in input_mods:
                    np.save(self.input_features[mod][base_filename],np.load(self.input_features[mod][base_filename])[:shortest_length])
                for i, mod in enumerate(input_mods):
                    length = np.load(self.input_features[mod][base_filename]).shape[0]
                    if i == 0:
                        length_0 = length
                    else:
                        assert length == length_0

            for mod in output_mods:
                feature_file = data_path.joinpath(base_filename+"."+mod+".npy")
                self.output_features[mod][base_filename] = feature_file

            if fix_lengths:
                shortest_length = 99999999999
                for mod in output_mods:
                    length = np.load(self.output_features[mod][base_filename]).shape[0]
                    if length < shortest_length:
                        shortest_length = length
                for mod in output_mods:
                    if mod not in input_mods:
                        np.save(self.output_features[mod][base_filename],np.load(self.output_features[mod][base_filename])[:shortest_length])
                for i, mod in enumerate(output_mods):
                    length = np.load(self.output_features[mod][base_filename]).shape[0]
                    if i == 0:
                        length_0 = length
                    else:
                        assert length == length_0

            self.base_filenames.append(base_filename)

        print("sequences added: "+str(len(self.base_filenames)))
        assert len(self.base_filenames)>0, "List of files for training cannot be empty"
        for mod in input_mods:
            assert len(self.input_features[mod].values()) == len(self.base_filenames)
        for mod in output_mods:
            assert len(self.output_features[mod].values()) == len(self.base_filenames)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--sampling_rate', default=44100, type=float)
        parser.add_argument('--input_modalities', default='mp3_mel_100')
        parser.add_argument('--output_modalities', default='mp3_mel_100')
        parser.add_argument('--input_lengths', help='input sequence length')
        parser.add_argument('--output_lengths', help='output sequence length')
        parser.add_argument('--output_time_offsets', default="1", help='time shift between the last read input, and the output predicted. The default value of 1 corresponds to predicting the next output')
        parser.add_argument('--input_time_offsets', default="0", help='time shift between the beginning of each modality and the first modality')
        parser.add_argument('--max_token_seq_len', type=int, default=1024)
        parser.set_defaults(output_length=1)
        parser.set_defaults(output_channels=1)

        return parser

    def name(self):
        return "MultiModalDataset"

    def __getitem__(self, item):
        base_filename = self.base_filenames[item]

        input_lengths = self.input_lengths
        output_lengths = self.output_lengths
        output_time_offsets = self.output_time_offsets
        input_time_offsets = self.input_time_offsets

        input_mods = self.opt.input_modalities.split(",")
        output_mods = self.opt.output_modalities.split(",")

        input_features = []
        output_features = []

        for i, mod in enumerate(input_mods):
            input_feature = np.load(self.input_features[mod][base_filename])
            input_features.append(input_feature)

        for i, mod in enumerate(output_mods):
            output_feature = np.load(self.output_features[mod][base_filename])
            output_features.append(output_feature)

        x = [input_feature.transpose(1,0) for input_feature in input_features]
        y = [output_feature.transpose(1,0) for output_feature in output_features]

        # normalization of individual features for the sequence
        # not doing this any more as we are normalizing over all examples now
        #x = [(xx-np.mean(xx,-1,keepdims=True))/(np.std(xx,-1,keepdims=True)+1e-5) for xx in x]
        #y = [(yy-np.mean(yy,-1,keepdims=True))/(np.std(yy,-1,keepdims=True)+1e-5) for yy in y]

        ## we pad the song features with zeros to imitate during training what happens during generation
        #x = [np.concatenate((np.zeros(( xx.shape[0],max(0,max(output_time_offsets)) )),xx),1) for xx in x]
        #y = [np.concatenate((np.zeros(( yy.shape[0],max(0,max(output_time_offsets)) )),yy),1) for yy in y]
        ## we also pad at the end to allow generation to be of the same length of sequence, by padding an amount corresponding to time_offset
        #x = [np.concatenate((xx,np.zeros(( xx.shape[0],max(0,max(input_lengths)+max(input_time_offsets)-(min(output_time_offsets)+min(output_lengths)-1)) ))),1) for xx in x]
        #y = [np.concatenate((yy,np.zeros(( yy.shape[0],max(0,max(input_lengths)+max(input_time_offsets)-(min(output_time_offsets)+min(output_lengths)-1)) ))),1) for yy in y]

        ## WINDOWS ##
        # sample indices at which we will get opt.num_windows windows of the sequence to feed as inputs
            # TODO: make this deterministic, and determined by `item`, so that one epoch really corresponds to going through all the data..
        sequence_length = x[0].shape[-1]
        #indices = np.random.choice(range(0,sequence_length-max(max(input_lengths)+max(input_time_offsets),max(output_time_offsets)+max(output_lengths))),size=self.opt.num_windows,replace=True)
        #max_i = sequence_length-max(max(input_lengths)+max(input_time_offsets),max(output_time_offsets)+max(output_lengths))
        #indices = np.random.choice(range(0,20),size=self.opt.num_windows,replace=True)
        indices = np.random.choice([0,1,2,3,4],size=1)

        ## CONSTRUCT TENSOR OF INPUT FEATURES ##
        input_windows = [torch.tensor([xx[:,i+input_time_offsets[j]:i+input_time_offsets[j]+input_lengths[j]] for i in indices]).float() for j,xx in enumerate(x)]

        ## CONSTRUCT TENSOR OF OUTPUT FEATURES ##
        output_windows = [torch.tensor([yy[:,i+output_time_offsets[j]:i+output_time_offsets[j]+output_lengths[j]] for i in indices]).float() for j,yy in enumerate(y)]

        # print(input_windows.shape)

        # return {'input': input_windows, 'target': output_windows}
        return_dict = {}

        for i,mod in enumerate(input_mods):
            # print(input_windows[i].shape)
            return_dict["in_"+mod] = input_windows[i]
        for i,mod in enumerate(output_mods):
            return_dict["out_"+mod] = output_windows[i]

        return return_dict

    def __len__(self):
        return len(self.base_filenames)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
