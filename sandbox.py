import pickle

thing = pickle.load(open("data/motions/gWA_sBM_cAll_d26_mWA1_ch08.pkl", "rb"))

thing['smpl_poses'].shape
thing['smpl_trans'].shape

import numpy as np

audio_feats = np.load("data/dev_audio/gHO_sBM_c01_d19_mHO3_ch02.mp3_mel_100.npy")
audio_feats = np.load("data/dev_audio/gBR_sBM_c01_d05_mBR5_ch09.mp3_mel_100.npy")

audio_feats.shape
max(1,2)
