#!/bin/bash

# export PYTHONPATH="/home_directory/.local/lib/python3.5/site-packages/"

#py=python3
py=python
dataset=multimodal
model=transformer
#exp=aistpp_big
exp=aistpp_fixed

$py scripts/training/train.py --data_dir=./data/scaled_features --dataset_name=$dataset --model=$model --batch_size=20 --num_windows=2 --nepoch=500 --nepoch_decay=500 \
    --print_freq=10 --experiment_name=$exp --save_by_iter --save_latest_freq=5000 --checkpoints_dir scripts/training\
    --dins="219,103" \
    --douts="219" \
    --input_modalities="joint_angles_scaled,mel_ddcpca_scaled" \
    --output_modalities="joint_angles_scaled" \
    --input_lengths="140,240" \
    --output_lengths="20" \
    --output_time_offset="121" \
    --predicted_inputs="20,0" \
    --nlayers=12 \
    --nhead=10 \
    --dhid=800 \
    --val_epoch_freq=0 \
    --gpu_ids=0 \
    --workers=4 \
    --dropout=0 \
    --continue_train \
    --load_iter=1415000 \
    --learning_rate=0.00005 \
