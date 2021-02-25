#!/bin/bash

# export PYTHONPATH="/home_directory/.local/lib/python3.5/site-packages/"

#py=python3
py=python
dataset=multimodal
model=transformer
#exp=aistpp_big
exp=aistpp_short

$py scripts/training/train.py --data_dir=./data --dataset_name=$dataset --model=$model --batch_size=16 --num_windows=10 --nepoch=1000 --nepoch_decay=1000 \
    --print_freq=10 --experiment_name=$exp --save_by_iter --save_latest_freq=5000 --checkpoints_dir scripts/training\
    --din=$((219+512)) \
    --dout=$((219)) \
    --input_modalities="pkl_joint_angles_mats,mp3_multi_mel_80.npy_ddc_hidden" \
    --output_modalities="pkl_joint_angles_mats" \
    --input_seq_len=128 \
    --lr 0.00002 \
    --nlayers=12 \
    --nhead=10 \
    --d_model=800 \
    --dhid=800 \
    --output_seq_len=20 \
    --output_time_offset=109 \
    --prefix_length=108 \
    --val_epoch_freq=0 \
    --gpu_ids=0 \
    --workers=4 \
    --continue_train \
    --load_iter=240000 \
