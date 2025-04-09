
import os
import tyro
import numpy as np
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt

def main(gt_file: Path):
    gt_map = pkl.load(open(gt_file, 'rb'))
    

    data = gt_map['data'].astype(np.float32).T
    data[data < 0] = 50
    data /= 100.0
    plt.imshow(data, cmap='bone')
    plt.colorbar()
    plt.title("Ground Truth Map")
    plt.show()



if __name__ == '__main__':
    tyro.cli(main)