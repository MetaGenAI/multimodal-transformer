#!/bin/bash

# export PYTHONPATH="/home_directory/.local/lib/python3.5/site-packages/"

#py=python3
py=python
dataset=multimodal
model=transformer
exp=aistpp_big

$py scripts/training/train.py --data_dir=./data_sample --dataset_name=$dataset --model=$model --batch_size=4 --num_windows=10 --nepoch=500 --nepoch_decay=500 \
    --print_freq=10 --experiment_name=$exp --save_by_iter --save_latest_freq=5000 --checkpoints_dir scripts/training\
    --dins="219,512" \
    --douts="219" \
    --input_modalities="pkl_joint_angles_mats,mp3_multi_mel_80.npy_ddc_hidden" \
    --output_modalities="pkl_joint_angles_mats" \
    --input_lengths="140,240" \
    --output_lengths="20" \
    --output_time_offset="121" \
    --predicted_inputs="20,0" \
    --nlayers=16 \
    --nhead=15 \
    --d_model=1500 \
    --dhid=1500 \
    --val_epoch_freq=0 \
    --gpu_ids=0 \
    --workers=4 \
    #--continue_train \
    #--load_iter=400000 \
