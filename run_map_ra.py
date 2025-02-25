import os
import tyro
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join as pjoin
from pathlib import Path

import jax
import jax.numpy as jnp

from mmwcas.dataset import ChirpPoseDataset
from sar import RAmapping, RadarMap

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_sar(
    folder: str,
    load: str = None,
    name: str = "current_time",
    save_video: bool = False,
    map_extent: float = 10.0,
    resolution_scale: int = 2,
    r: str = "radar0",
) -> None:

    data_dir = Path(folder)
    seq_name = data_dir.name
    name = str(int(time.time())) if name == "current_time" else name
    save_dir = f"ramap/{seq_name}_{name}"
    os.makedirs(f"{save_dir}", exist_ok=True)

    dataset = ChirpPoseDataset(pjoin(data_dir, r))

    # ramapping = RAmapping(dataset, map_extent)
    mapper = RAmapping(dataset, map_extent, resolution_scale)
    map_val = mapper.map_val
    ramapping = jax.jit(mapper.__call__)

    if save_video:
        writer = imageio.get_writer(f"{save_dir}/mapping.mp4", fps=dataset.fps)
        color_map = plt.get_cmap("hot")

    for chirps, poses, stamp in tqdm(dataset):
        pose = poses[0, 4]  # center: chirp 0 tx 4
        map_val = ramapping(pose, chirps, map_val)

        p = 1.0 - 1.0 / (1.0 + np.exp(map_val))
        img = color_map(p)[..., :3] * 255

        # ra = np.clip(ra, 0, 1)
        # img = color_map(ra)[..., :3] * 255

        writer.append_data(img.astype(np.uint8))


if __name__ == "__main__":
    cli = tyro.cli(run_sar)
