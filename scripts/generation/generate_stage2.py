import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

sys.path.append(ROOT_DIR)
import argparse
import time
from models import create_model
import json, pickle
import torch
import numpy as np
import models.constants as constants
from math import ceil
from scipy import signal

from scripts.generation.level_generation_utils import extract_features, make_level_from_notes, get_notes_from_stepmania_file

parser = argparse.ArgumentParser(description='Generate Beat Saber level from song')
parser.add_argument('--song_path', type=str)
parser.add_argument('--json_file', type=str)
parser.add_argument('--experiment_name', type=str)
parser.add_argument('--checkpoint', type=str, default="latest")
parser.add_argument('--temperature', type=float, default=1.00)
parser.add_argument('--bpm', type=float, default=None)
parser.add_argument('--generate_full_song', action="store_true")
parser.add_argument('--use_beam_search', action="store_true")
parser.add_argument('--open_in_browser', action="store_true")
parser.add_argument('--cuda', action="store_true")

args = parser.parse_args()

experiment_name = args.experiment_name+"/"
checkpoint = args.checkpoint
temperature=args.temperature
song_path=args.song_path
json_file=args.json_file

from pathlib import Path
song_name = Path(song_path).stem

print("STAGE TWO!")
#%%
''' LOAD MODEL, OPTS, AND WEIGHTS (for stage1 if two_stage) '''

#loading opt object from experiment, and constructing Struct object after adding some things
opt = json.loads(open("scripts/training/"+experiment_name+"opt.json","r").read())
# extra things Beam search wants
if args.cuda:
    opt["gpu_ids"] = [0]
else:
    opt["gpu_ids"] = []
opt["checkpoints_dir"] = "scripts/training"
opt["load_iter"] = int(checkpoint)
if args.cuda:
    opt["cuda"] = True
else:
    opt["cuda"] = False
opt["batch_size"] = 1
opt["beam_size"] = 20
opt["n_best"] = 1
# opt["using_bpm_time_division"] = True
opt["continue_train"] = False
# opt["max_token_seq_len"] = len(notes)
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
opt = Struct(**opt)

model = create_model(opt)
model.setup()
checkpoint = "iter_"+checkpoint
model.load_networks(checkpoint)

''' GET SONG FEATURES for stage two '''
#%%

seq_id="gLH_sBM_cAll_d16_mLH1_ch04"
#seq_id="gWA_sBM_cAll_d26_mWA1_ch10"

sf = np.load("data/features/"+seq_id+".mp3_mel_ddcpca.npy")
#sound_features = np.load("lou_bega_mambolovania_-250249188128876949.mp3_multi_mel_80.npy_ddc_hidden.npy")
mf = np.load("data/features/"+seq_id+".pkl_joint_angles_mats.npy")
mf_mean=np.mean(mf,0,keepdims=True)
mf_std = np.std(mf,0,keepdims=True)+1e-5
sf = (sf-np.mean(sf,0,keepdims=True))/(np.std(sf,0,keepdims=True)+1e-5)
mf = (mf-mf_mean)/(mf_std)


sf = sf[:1024]
mf = mf[:1024]

#motion_features = np.zeros((sound_features.shape[0],219))

# features = np.concatenate([motion_features,sound_features],1)
# features = features.transpose(1,0)
# print(features.shape)
features = {}
features["in_pkl_joint_angles_mats"] = np.expand_dims(np.expand_dims(mf.transpose(1,0),0),0)
features["in_mp3_mel_ddcpca"] = np.expand_dims(np.expand_dims(sf.transpose(1,0),0),0)

predicted_modes = model.generate(features)

predicted_modes = (predicted_modes[0].cpu().numpy()*mf_std + mf_mean)

print(predicted_modes)

np.save(seq_id+".pkl_joint_angles_mats.generated.test.npz",predicted_modes)
#np.save("mambolovania",predicted_modes.cpu().numpy())
