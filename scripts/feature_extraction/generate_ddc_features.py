import numpy as np
from pathlib import Path
import json
import os.path
import sys
import argparse
import time
from models import create_model
import json, pickle
import torch
from math import ceil
from scipy import signal

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)
sys.path.append(ROOT_DIR)

from scripts.generation.level_generation_utils import extract_features, make_level_from_notes, get_notes_from_stepmania_file
import models.constants as constants
from feature_extraction import extract_features_hybrid, extract_features_mel, extract_features_multi_mel

parser = argparse.ArgumentParser(description='Get DDC features from song features')
parser.add_argument("data_path", type=str, help="Directory contining Beat Saber level folders")
parser.add_argument('--checkpoints_dir', type=str)
parser.add_argument('--experiment_name', type=str)
parser.add_argument('--peak_threshold', type=float, default=0.0148)
parser.add_argument('--checkpoint', type=str, default="latest")
parser.add_argument('--temperature', type=float, default=1.00)
parser.add_argument('--cuda', action="store_true")

# parser.add_argument("--feature_name", metavar='', type=str, default="mel", help="mel, chroma, multi_mel")
# parser.add_argument("--feature_size", metavar='', type=int, default=100)
# parser.add_argument("--sampling_rate", metavar='', type=float, default=44100.0)
parser.add_argument("--step_size", metavar='', type=float, default=0.01666666666)
parser.add_argument("--replace_existing", action="store_true")

args = parser.parse_args()

# makes arugments into global variables of the same name, used later in the code
globals().update(vars(args))
data_path = Path(data_path)

## distributing tasks accross nodes ##
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)
# print("creating {} of size {}".format(feature_name, feature_size))

experiment_name = args.experiment_name+"/"
checkpoint = args.checkpoint
temperature=args.temperature

from pathlib import Path

''' LOAD MODEL, OPTS, AND WEIGHTS'''
#%%

##loading opt object from experiment
opt = json.loads(open(ROOT_DIR.__str__()+"/scripts/training/"+experiment_name+"opt.json","r").read())
# we assume we have 1 GPU in generating machine :P
if args.cuda:
    opt["gpu_ids"] = [0]
else:
    opt["gpu_ids"] = []
opt["checkpoints_dir"] = args.checkpoints_dir
opt["load_iter"] = int(checkpoint)
if args.cuda:
    opt["cuda"] = True
else:
    opt["cuda"] = False
opt["experiment_name"] = args.experiment_name.split("/")[0]
if "dropout" not in opt: #for older experiments
    opt["dropout"] = 0.0
# construct opt Struct object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
opt = Struct(**opt)

assert opt.binarized

model = create_model(opt)
model.setup()
receptive_field = 1

checkpoint = "iter_"+checkpoint
model.load_networks(checkpoint)

from scipy.signal import resample
from scipy.interpolate import interp1d

def ResampleLinear1D(original, targetLen):
    index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int) #Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * np.expand_dims(np.expand_dims(1.0-index_rem,1),1) + val2 * np.expand_dims(np.expand_dims(index_rem,1),1)
    assert(len(interp) == targetLen)
    return interp


#assuming mp3 for now. TODO: generalize
candidate_feature_files = sorted(data_path.glob('**/*mp3_multi_mel_80.npy'), key=lambda path: path.parent.__str__())
num_tasks = len(candidate_feature_files)
num_tasks_per_job = num_tasks//size
tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))
if rank < num_tasks%size:
    tasks.append(size*num_tasks_per_job+rank)

for i in tasks:
    path = candidate_feature_files[i]
    features_file = str(path)+"_"+"ddc_hidden"+".npy"
    sr = opt.sampling_rate
    hop = int(opt.step_size*sr)
    features = np.load(path)

    #generate level
    # first_samples basically works as a padding, for the first few outputs, which don't have any "past part" of the song to look at.
    first_samples = torch.full((1,opt.output_channels,receptive_field//2),constants.START_STATE,dtype=torch.float)
    features, peak_probs = model.generate_features(features)
    peak_probs = peak_probs[0,:,-1].cpu().detach().numpy()
    features = features.cpu().detach().numpy()
    features = np.transpose(ResampleLinear1D(np.transpose(features,(1,0,2)),int(np.floor(features.shape[1]*0.01/0.016666666))),(1,0,2))[0,1:,:]
    print(features.shape)
    np.save(features_file,features)
    window = signal.hamming(ceil(constants.HUMAN_DELTA/opt.step_size))
    smoothed_peaks = np.convolve(peak_probs,window,mode='same')

    thresholded_peaks = smoothed_peaks*(smoothed_peaks>args.peak_threshold)
    peaks = signal.find_peaks(thresholded_peaks)[0]
    print("number of peaks", len(peaks))
