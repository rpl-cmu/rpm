#!/bin/zsh

seq_list=(
    c_corridor
    "nsh_a1"
    "square1"
    "square2"
    "nsh"
    "nsh_b"
    "nsh_short"
    "z_shape1"
    "z_shape2"
    "wean"
    # "tepper"
    # "nsh_a"
    # "cic"
)

# Loop through each sequence in the list
for seq in "${seq_list[@]}"; do
    # Create a directory for the sequence
    echo $seq

    python eval_cd.py \
    --gt_file ~/data/sftp/$seq/map/lidar.pkl \
    --sar_file exps_pc/map_sar/$seq\_test/prob.pkl \
    --ra_file exps_pc/map_ra/$seq\_test/prob.pkl \
    --cfar_file ~/data/sftp/$seq/map/cfar.pkl \
    --output exps_pc/map_eval/$seq\.pdf
done