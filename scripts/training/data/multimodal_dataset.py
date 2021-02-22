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

        self.input_features = {input_mod:{} for input_mod in input_mods}
        self.output_features = {output_mod:{} for output_mod in output_mods}

        input_seq_len = self.opt.input_seq_len
        output_seq_len = self.opt.output_seq_len
        min_length = max(input_seq_len, self.opt.output_time_offset + output_seq_len) - min(0,self.opt.output_time_offset)
        print(min_length)

        #Get the list of files containing features (in numpy format for now), and populate the dictionaries of input and output features (separated by modality)
        for base_filename in temp_base_filenames:
            file_too_short = False
            for i, mod in enumerate(input_mods):
                feature_file = data_path.joinpath("features").joinpath(base_filename+"."+mod+".npy")
                # print(feature_file)
                try:
                    features = np.load(feature_file)
                    length = features.shape[0]
                    # print(features.shape)
                    # print(length)
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
                feature_file = data_path.joinpath("features").joinpath(base_filename+"."+mod+".npy")
                try:
                    features = np.load(feature_file)
                    length = features.shape[0]
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
                feature_file = data_path.joinpath("features").joinpath(base_filename+"."+mod+".npy")
                self.input_features[mod][base_filename] = feature_file
            # shortest_length = 99999999999
            # for mod in input_mods:
            #     length = np.load(self.input_features[mod][base_filename]).shape[0]
            #     if length < shortest_length:
            #         shortest_length = length
            # for mod in input_mods:
            #     np.save(self.input_features[mod][base_filename],np.load(self.input_features[mod][base_filename])[:shortest_length])
            # for i, mod in enumerate(input_mods):
            #     length = np.load(self.input_features[mod][base_filename]).shape[0]
            #     if i == 0:
            #         length_0 = length
            #     else:
            #         assert length == length_0

            for mod in output_mods:
                feature_file = data_path.joinpath("features").joinpath(base_filename+"."+mod+".npy")
                self.output_features[mod][base_filename] = feature_file
            # shortest_length = 99999999999
            # for mod in output_mods:
            #     length = np.load(self.output_features[mod][base_filename]).shape[0]
            #     if length < shortest_length:
            #         shortest_length = length
            # for mod in output_mods:
            #     if mod not in input_mods:
            #         np.save(self.output_features[mod][base_filename],np.load(self.output_features[mod][base_filename])[:shortest_length])
            # for i, mod in enumerate(output_mods):
            #     length = np.load(self.output_features[mod][base_filename]).shape[0]
            #     if i == 0:
            #         length_0 = length
            #     else:
            #         assert length == length_0
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
        parser.add_argument('--input_seq_len', type=int, default=512)
        parser.add_argument('--output_seq_len', type=int, default=512)
        parser.add_argument('--padded_length', type=int, default=3000000)
        parser.add_argument('--chunk_length', type=int, default=9000)
        # the input features at each time step consiste of the features at the time steps from now to time_shifts in the future
        parser.add_argument('--output_time_offset', type=int, default=1, help='time shift between the last read input, and the output predicted. The default value of 1 corresponds to predicting the next output')
        parser.add_argument('--flatten_context', action='store_true', help='whether to flatten the temporal context added for each time point into the feature dimension, or not')
        parser.add_argument('--max_token_seq_len', type=int, default=1024)
        parser.set_defaults(output_length=1)
        parser.set_defaults(output_channels=1)

        return parser

    def name(self):
        return "MultiModalDataset"

    def __getitem__(self, item):
        base_filename = self.base_filenames[item]

        input_length = self.opt.input_seq_len
        output_length = self.opt.output_seq_len
        time_offset = self.opt.output_time_offset

        input_mods = self.opt.input_modalities.split(",")
        output_mods = self.opt.output_modalities.split(",")

        input_features = None
        output_features = None
        input_mod_sizes = []
        output_mod_sizes = []

        for i, mod in enumerate(input_mods):
            if i==0:
                input_features = np.load(self.input_features[mod][base_filename])
                input_mod_sizes.append(input_features.shape[0])
            else:
                input_feature = np.load(self.input_features[mod][base_filename])
                input_mod_sizes.append(input_feature.shape[0])
                input_features = np.concatenate([input_features,input_feature], 1)

        for i, mod in enumerate(output_mods):
            if i==0:
                output_features = np.load(self.output_features[mod][base_filename])
                output_mod_sizes.append(output_features.shape[0])
            else:
                output_feature = np.load(self.output_features[mod][base_filename])
                output_mod_sizes.append(output_feature.shape[0])
                output_features = np.concatenate([output_features,output_feature], 1)

        x = input_features.transpose(1,0)
        y = output_features.transpose(1,0)
        # print(x.shape)

        # we pad the song features with zeros to imitate during training what happens during generation
        if len(x.shape) == 2: # one feature dimension in y
            x = np.concatenate((np.zeros(( x.shape[0],max(0,time_offset) )),x),1)
            y = np.concatenate((np.zeros(( y.shape[0],max(0,time_offset) )),y),1)
            # we also pad at the end to allow generation to be of the same length of sequence, by padding an amount corresponding to time_offset
            x = np.concatenate((x,np.zeros(( x.shape[0],max(0,input_length-(time_offset+output_length-1)) ))),1)
            y = np.concatenate((y,np.zeros(( y.shape[0],max(0,input_length-(time_offset+output_length-1)) ))),1)

        ## WINDOWS ##
        # sample indices at which we will get opt.num_windows windows of the sequence to feed as inputs
            # TODO: make this deterministic, and determined by `item`, so that one epoch really corresponds to going through all the data..
        sequence_length = x.shape[-1]
        indices = np.random.choice(range(0,sequence_length-max(input_length,time_offset+output_length)),size=self.opt.num_windows,replace=True)
        # print(indices)

        ## CONSTRUCT TENSOR OF INPUT FEATURES ##
        input_windows = [x[:,i:i+input_length] for i in indices]
        input_windows = torch.tensor(input_windows)
        # input_windows = (input_windows - input_windows.mean())/torch.abs(input_windows).max()

        ## CONSTRUCT TENSOR OF OUTPUT FEATURES ##
        output_windows = [y[:,i+time_offset:i+time_offset+output_length] for i in indices]
        output_windows = torch.tensor(output_windows)
        # output_windows = (output_windows - output_windows.mean())/torch.abs(output_windows).max()

        # print(input_windows.shape)

        return {'input': input_windows.float(), 'target': output_windows.float(), 'feature_sizes': (input_mod_sizes, output_mod_sizes)}

    def __len__(self):
        return len(self.base_filenames)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
