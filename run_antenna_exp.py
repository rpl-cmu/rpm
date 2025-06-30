#!/usr/bin/env python3
import subprocess
from pathlib import Path


seq_list = [
    # "c_corridor",
    # "nsh_a",
    # "nsh_a1",
    # "tepper",
    # "square1",
    # "square2",
    # "nsh",
    # "nsh_b",
    # "nsh_short",
    "z_shape1",
    "z_shape2",
    "wean",
    # "cic",
]

antenna_list = [
    "--name 1 --rx 0 1 2 3 --tx 0 1 2",
    "--name 2 --rx 0 1 2 3 4 5 6 7 --tx 0 1 2 3 4 5",
    "--name 3 --rx 0 1 2 3 4 5 6 7 8 9 10 11 --tx 0 1 2 3 4 5 6 7 8 9",
    "--name 4 --rx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --tx 0 1 2 3 4 5 6 7 8 9 10 11",
]

for seq in seq_list:
    # print(seq)
    for antenna in antenna_list:
        # print(antenna)
        folder = str(Path.home() / "data/sftp" / seq)

        antenna_args = antenna.split()

        cmd = [
            "python",
            "map_sar_dual.py",
            "--folder",
            folder,
            # "--save-video",
        ] + antenna_args

        print("Executing:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        