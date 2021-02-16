import numpy as np
import librosa
from pathlib import Path
import json
import os.path
import sys
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)
sys.path.append(ROOT_DIR)
from feature_extraction import extract_features_hybrid, extract_features_mel, extract_features_multi_mel

parser = argparse.ArgumentParser(description="Preprocess songs data")

parser.add_argument("data_path", type=str, help="Directory contining Beat Saber level folders")
parser.add_argument("--feature_name", metavar='', type=str, default="mel", help="mel, chroma, multi_mel")
parser.add_argument("--feature_size", metavar='', type=int, default=100)
parser.add_argument("--sampling_rate", metavar='', type=float, default=44100.0)
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
print("creating {} of size {}".format(feature_name, feature_size))

#assuming mp3 for now. TODO: generalize
candidate_audio_files = sorted(data_path.glob('**/*.mp3'), key=lambda path: path.parent.__str__())
num_tasks = len(candidate_audio_files)
num_tasks_per_job = num_tasks//size
tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))
if rank < num_tasks%size:
    tasks.append(size*num_tasks_per_job+rank)

for i in tasks:
    path = candidate_audio_files[i]
    song_file_path = path.__str__()
    # feature files are going to be saved as numpy files
    features_file = song_file_path+"_"+feature_name+"_"+str(feature_size)+".npy"

    if replace_existing or not os.path.isfile(features_file):
        print("creating feature file",i)

        # get song
        y_wav, sr = librosa.load(song_file_path, sr=sampling_rate)

        sr = sampling_rate
        hop = int(sr * step_size)

        #get feature
        if feature_name == "chroma":
            features = extract_features_hybrid(y_wav,sr,hop)
        elif feature_name == "mel":
            features = extract_features_mel(y_wav,sr,hop,mel_dim=feature_size)[:,1:]
        elif feature_name == "multi_mel":
            features = extract_features_multi_mel(y_wav, sr=sampling_rate, hop=hop, nffts=[1024,2048,4096], mel_dim=feature_size)

        np.save(features_file,features)
