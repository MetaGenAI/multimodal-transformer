#!/bin/bash

# song_path=$1

exp2=aistpp_fixed2
cpt2=55000
#cpt2=1200000
#cpt2=1450000
#exp2=test_block_selection
#cpt2=$3
#ddc_file=/home/guillefix/ddc_infer/57257860-f345-4e5c-ba69-36f57b561118/57257860-f345-4e5c-ba69-36f57b561118.sm

py=python

#mkdir generated

$py scripts/generation/generate_stage2.py --cuda --song_path . --experiment_name $exp2 --checkpoint $cpt2 \
#$py scripts/generation/generate_stage2.py --cuda --song_path $song_path --experiment_name $exp2 --checkpoint $cpt2 \
#    --temperature 1.00 \
#    --use_beam_search \
