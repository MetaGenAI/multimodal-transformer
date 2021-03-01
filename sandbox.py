#### looking at the data
import pickle
import matplotlib.pyplot as plt
from pymo.rotation_tools import unroll_1, unroll_2

thing = pickle.load(open("data/motions/gBR_sBM_cAll_d06_mBR4_ch10.pkl", "rb"))
thing = pickle.load(open("data/motions/gWA_sFM_cAll_d26_mWA1_ch09.pkl", "rb"))
thing = pickle.load(open("/home/guillefix/code/mocap/PyMO/demos/data/gBR_sBM_cAll_d04_mBR0_ch01.pkl", "rb"))
thing = pickle.load(open("/home/guillefix/Downloads/PyMO/demos/data/gBR_sBM_cAll_d04_mBR0_ch01.pkl", "rb"))
thing = pickle.load(open("data/motions/gBR_sBM_cAll_d04_mBR0_ch01.pkl", "rb"))
thing = pickle.load(open("/home/guillefix/Downloads/motions/gBR_sBM_cAll_d04_mBR0_ch01.pkl", "rb"))
thing = pickle.load(open("data/motions/gJB_sFM_cAll_d09_mJB1_ch21.pkl", "rb"))
thing = pickle.load(open("/home/guillefix/Downloads/aist_plusplus_final_motions_gBR_sBM_cAll_d04_mBR0_ch01.pkl", "rb"))

thing['smpl_poses'].shape
thing.keys()
poses = thing['smpl_poses']
poses = poses.reshape(-1,24*3)
# unrolled_poses = unroll_1(poses.reshape(-1,3))
rots = poses[:,:3]
for i in range(24):
    poses[:,i*3:(i+1)*3] = unroll_2(poses[:,i*3:(i+1)*3])
unrolled_poses.shape
poses = unrolled_poses.reshape(-1,24*3)
poses[593:693,:3]
poses[593:693,:3][38:]

poses[:300,:3]
import numpy as np
np.diff(poses[90:110,:3], axis=0)
np.diff(poses[90:110,:3], axis=0)

plt.matshow(poses[:300,:3])
# plt.matshow(poses[1000:1500,:3])
plt.matshow(poses[:300,:3])

thing

thing['smpl_poses'].shape
thing['smpl_trans'].shape


angles = [[euler_angles for euler_angles in np.array(joint_angles).reshape(-1,3)] for joint_angles in thing['smpl_poses']]
features = get_rot_matrices(thing['smpl_poses'])

from scipy.spatial.transform import Rotation as R

def get_rot_matrices(joint_traj):
    return np.stack([np.concatenate([R.from_euler('xyz',euler_angles).as_matrix().flatten() for euler_angles in np.array(joint_angles).reshape(-1,3)]) for joint_angles in joint_traj])

def get_features(motion_data):
    joint_angle_feats = get_rot_matrices((motion_data['smpl_poses']))
    return np.concatenate([joint_angle_feats,motion_data['smpl_trans']],1)

import numpy as np

audio_feats = np.load("data/features/gWA_sBM_c03_d26_mWA1_ch08.mp3_mel_100.npy")
audio_feats = np.load("data/features/gWA_sBM_c01_d26_mWA1_ch08.mp3_mel_100.npy")
audio_feats = np.load("data/dev_audio/gBR_sBM_c01_d06_mBR4_ch10.mp3_mel_100.npy")
audio_feats = np.load("data/dev_audio/gBR_sBM_c01_d06_mBR2_ch02.mp3_mel_100.npy")
# audio_feats = np.load("data/dev_audio/gHO_sBM_c01_d19_mHO3_ch08.mp3_multi_mel_80.npy_ddc_hidden.npy")
audio_feats = np.load("data/features/gHO_sBM_cAll_d19_mHO3_ch08.mp3_multi_mel_80.npy_ddc_hidden.npy")
audio_feats = np.load("data/features/gWA_sFM_cAll_d26_mWA1_ch09.mp3_mel_ddcpca.npy")
motion_feats = np.load("data/features/gWA_sFM_cAll_d26_mWA1_ch09.pkl_joint_angles_mats.npy")



Audio_feats.shape

audio_feats.shape
plt.matshow(motion_feats[100:200,:])

plt.matshow(audio_feats[100:200,:])

from sklearn import decomposition

pca = decomposition.PCA(n_components=512)

x = pca.fit_transform(audio_feats)
x.shape

pca.transform(x[:5])

plt.matshow(x[100:200,:2])

audio_feats.shape
max(1,2)

###########################
#playing with masks

import torch

sz=20
mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
mask[:3,:3]
mask[:,:10] = 1
plt.imshow(mask)
mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

###########################
#plotting masks hmm
#mega hacky lel

import numpy as np
import matplotlib.pyplot as plt
mask = np.load("4db8086b-ae2c-4d60-b6bb-720214f36ef0.np.npy")
mask[0][120:140]
mask[0][:140]
import torch
maskt = torch.from_numpy(mask)

ps = torch.nn.functional.softmax(maskt,dim=-1)
sm = lambda x: torch.nn.functional.softmax(torch.from_numpy(x),dim=-1)
ps[0]
mask[1].max()
mask[1].max()
mask.shape
mask[8]
mask[6].min()
plt.matshow(mask[8][120:140])
plt.matshow(sm(mask[9][120:140]))
mask[8][120:121]
plt.matshow(mask[9])
plt.matshow(mask[5])
plt.matshow(mask[9][0:1])


###########################
#audio
import librosa

y_wav, sr = librosa.load("data/dev_audio/gBR_sBM_c01_d06_mBR5_ch10.mp3", sr=48000)
y_wav
envelope = librosa.onset.onset_strength(y=y_wav,hop_length=480)

(y_wav.shape[0]/48000)/0.01

envelope.shape


###################
##

seq_id="gWA_sFM_cAll_d27_mWA2_ch17"
# seq_id="mambo"

# sf = np.load("data/features/"+seq_id+".mel_ddcpca_scaled.npy")
sf = np.load("test_data/"+seq_id+".mel_ddcpca_scaled.npy")
# mf = np.load("data/features/"+seq_id+".joint_angles_scaled.npy")
mf = np.load("test_data/"+seq_id+".joint_angles_scaled.npy")

sf.shape
sf
plt.matshow(sf)
