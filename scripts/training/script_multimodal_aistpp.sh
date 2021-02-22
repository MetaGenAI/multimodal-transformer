#!/bin/bash

# export PYTHONPATH="/home_directory/.local/lib/python3.5/site-packages/"

#py=python3
py=python
dataset=multimodal
model=transformer
exp=aistpp1

$py scripts/training/train.py --data_dir=./data --dataset_name=$dataset --model=$model --batch_size=2 --num_windows=3 --nepoch=500 --nepoch_decay=500 \
    --print_freq=10 --experiment_name=$exp --save_by_iter --save_latest_freq=500 --checkpoints_dir scripts/training\
    --din=$((219+512)) \
    --dout=$((219)) \
    --input_modalities="pkl_joint_angles_mats,mp3_multi_mel_80.npy_ddc_hidden" \
    --output_modalities="pkl_joint_angles_mats" \
    --input_seq_len=512 \
    --output_seq_len=20 \
    --output_time_offset=493 \
    --prefix_length=492 \
    --val_epoch_freq=0 \
    --gpu_ids=0 \
    --workers=0 \
    # --continue_train \
    # --load_iter=1600000 \
