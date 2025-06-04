import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
import pickle as pkl
import tyro
from pathlib import Path

def main(file_name : Path):
    file_extension = file_name.suffix

    map_data = None
    if (file_extension == ".pkl"):
        map_data = pkl.load(open(file_name, 'rb'))['data']
    elif (file_extension == ".png"):
        map_data = iio.imread(file_name).astype(np.float32)
        map_data = (map_data / 3).sum(axis=2)
    else:
        print("Map file format unsupported")
        return
    
    plt.imshow(map_data)
    plt.show()

if __name__ == "__main__":
    tyro.cli(main)