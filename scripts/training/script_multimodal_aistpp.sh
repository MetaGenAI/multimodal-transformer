#!/bin/bash

# export PYTHONPATH="/home_directory/.local/lib/python3.5/site-packages/"

#py=python3
py=python
dataset=multimodal
model=transformer
#exp=aistpp_big
exp=aistpp_212_mfcc

$py scripts/training/train.py --data_dir=./data --dataset_name=$dataset --model=$model --batch_size=20 --num_windows=2 --nepoch=100 --nepoch_decay=500 \
    --print_freq=10 --experiment_name=$exp --save_by_iter --save_latest_freq=5000 --checkpoints_dir scripts/training\
    --lr=0.0001 \
    --dins="219,102" \
    --douts="219" \
    --input_modalities="pkl_joint_angles_mats,mp3_mel_ddcpca" \
    --output_modalities="pkl_joint_angles_mats" \
    --input_lengths="140,240" \
    --output_lengths="20" \
    --output_time_offset="121" \
    --predicted_inputs="20,0" \
    --nlayers=12 \
    --nhead=10 \
    --dhid=800 \
    --val_epoch_freq=0 \
    --tpu_ids=0 \
    --workers=4 \
    --dropout=0 \
    #--continue_train \
    #--load_iter=30000 \
