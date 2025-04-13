
import os
import tyro
import numpy as np
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt

def main(gt_file: Path, pred_file: Path):
    gt_map = pkl.load(open(gt_file, 'rb'))
    gt_data = gt_map['data'].astype(np.float32).T
    gt_data[gt_data < 0] = 50
    gt_data /= 100.0
    
    pred_map = pkl.load(open(pred_file, 'rb'))
    pred_data = pred_map['data'].astype(np.float32)

    print(gt_map)
    print(pred_map)

    plt.imsave("occusar/test_gt.png", 1-gt_data, cmap='bone')
    plt.imsave("occusar/test_pred.png", 1-pred_data, cmap='bone')

    

    # plt.imshow(data, cmap='bone')
    # plt.colorbar()
    # plt.title("Ground Truth Map")
    # plt.show()



if __name__ == '__main__':
    tyro.cli(main)