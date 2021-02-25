import librosa
import numpy as np
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
# parser.add_argument("--ddc_pca", action="store_true")


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

from sklearn import decomposition
pca = decomposition.PCA(n_components=512)
for i in tasks:
    path = candidate_audio_files[i]
    audio_file = path.__str__()
    ddc_features_file = audio_file+"_"+"ddc_hidden"+".npy"
    mfcc_features_file = audio_file+"_"+feature_name+"_"+str(feature_size)+".npy"
    features_file = audio_file+"_mel_ddcpca.npy"
    mfcc_features = np.load(mfcc_features_file).transpose(1,0)
    ddc_features = np.load(ddc_features_file)
    x = pca.fit_transform(ddc_features)
    features = np.cat([mfcc_features,x[:,:2]],1)
    np.save(features_file,features)
