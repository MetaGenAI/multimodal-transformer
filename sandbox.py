import pickle

thing = pickle.load(open("data/motions/gBR_sBM_cAll_d06_mBR4_ch10.pkl", "rb"))
thing = pickle.load(open("data/motions/gBR_sBM_cAll_d06_mBR2_ch02.pkl", "rb"))

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

audio_feats.shape

audio_feats.shape
max(1,2)
