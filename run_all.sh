#!/bin/zsh


seq_list=(
    # c_corridor
    # "nsh_a"
    # "nsh_a1"
    # "tepper"
    # "square1"
    # "square2"
    # "nsh"
    # "nsh_b"
    # "nsh_short"
    # "w_corridor1"
    # "w_corridor2"
    # "wean"
    "cic"
)

# Loop through each sequence in the list
for seq in "${seq_list[@]}"; do
    # Create a directory for the sequence
    echo $seq
    
    python map_sar_dual.py --folder ~/data/sftp/$seq  --save-video --name test
    python map_ra_dual.py --folder ~/data/sftp/$seq --name test --save_video
    # python eval_pc.py --gt_file ~/data/sftp/$seq/map/lidar.pkl --eval_file ~/data/sftp/$seq/map/cfar.pkl
    
done