import tyro
from glob import glob
from typing import Optional, Tuple, Dict
from os.path import join as pjoin

from mmwcas.dataset import CascadeADCDataset, RadarParam
from mmwcas.process import RadarCubeProc

import jax
import jax.numpy as jnp


def main(folder: str, radar_name: str, afft_size: Optional[Tuple[int, int]] = None):

    dataset = CascadeADCDataset(pjoin(folder, radar_name))
    process = jax.jit(RadarCubeProc(dataset.param, afft_size))

    for data in dataset:
        cube = process(jnp.asarray(data))
        print(cube.shape)


if __name__ == "__main__":
    tyro.cli(main, description="Run Radar Cube Processing")
