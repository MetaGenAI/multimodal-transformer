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
from scripts.feature_extraction.feature_extraction import extract_features_hybrid, extract_features_mel, extract_features_multi_mel

parser = argparse.ArgumentParser(description="Preprocess songs data")

parser.add_argument("data_path", type=str, help="Directory contining Beat Saber level folders")
parser.add_argument("--feature_name", metavar='', type=str, default="mel", help="mel, chroma, multi_mel")
parser.add_argument("--feature_size", metavar='', type=int, default=100)
parser.add_argument("--sampling_rate", metavar='', type=float, default=44100.0)
parser.add_argument("--step_size", metavar='', type=float, default=0.01)
parser.add_argument("--replace_existing", action="store_true")

args = parser.parse_args()

# makes arugments into global variables of the same name, used later in the code
globals().update(vars(args))
data_path = Path(data_path)

from scipy.spatial.transform import Rotation as R

def get_rot_matrices(joint_traj):
    return np.stack([np.concatenate([R.from_euler('xyz',euler_angles).as_matrix().flatten() for euler_angles in np.array(joint_angles).reshape(-1,3)]) for joint_angles in joint_traj])

def get_features(motion_data):
    joint_angle_feats = get_rot_matrices((motion_data['smpl_poses']))
    return np.concatenate([joint_angle_feats,motion_data['smpl_trans']],1)

## distributing tasks accross nodes ##
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)
print("creating {} of size {}".format(feature_name, feature_size))

candidate_motion_files = sorted(data_path.glob('**/*.pkl'), key=lambda path: path.parent.__str__())
num_tasks = len(candidate_audio_files)
num_tasks_per_job = num_tasks//size
tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))
if rank < num_tasks%size:
    tasks.append(size*num_tasks_per_job+rank)

for i in tasks:
    path = candidate_motion_files[i]
    motion_file_path = path.__str__()
    features_file = motion_file_path+"_"+"joint_angles_mats"+".npy"
    if replace_existing or not os.path.isfile(features_file):
        motion_data = pickle.load(open(path,"rb"))
        features = get_features(motion_data)
        print(features.shape)
        np.save(features_file,features)
