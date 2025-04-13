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
    map_extent: float = 10,
    resolution: float = 0.1,
    protect_range: float = 0.3,
    angle_fov: float = 20.0,
    amp_sigma: float = 2.0,
    prob_hit: float = 1,
    prob_miss: float = 0.45,
    clamp_log_max: float = 3.5,  # 0.97
    clamp_log_min: float = -2.0,  # 0.12
    angele_fft_size: int = 256,
    r: str = "radar0",
) -> None:

    data_dir = Path(folder)
    seq_name = data_dir.name
    name = str(int(time.time())) if name == "current_time" else name
    save_dir = f"exps/map_ra/{seq_name}_{name}"
    os.makedirs(f"{save_dir}", exist_ok=True)

    dataset = ChirpPoseDataset(pjoin(data_dir, r))

    mapper = RAmapping(
        dataset=dataset,
        map_extent=map_extent,
        protect_range=protect_range,
        resolution=resolution,
        angle_3dB=angle_fov,
        amp_sigma=amp_sigma,
        prob_hit=prob_hit,
        prob_miss=prob_miss,
        clamp_log_max=clamp_log_max,
        clamp_log_min=clamp_log_min,
        angele_fft_size=angele_fft_size,
    )
    log_odds = mapper.map_val
    ramapping = jax.jit(mapper.__call__)

    if save_video:
        writer = imageio.get_writer(f"{save_dir}/mapping.mp4", fps=dataset.fps)
        color_map = plt.get_cmap("bone")

    for chirps, poses, stamp in tqdm(dataset):
        pose = poses[0, 4]  # center: chirp 0 tx 4
        log_odds = ramapping(pose, chirps, log_odds)

        p = 1.0 - 1.0 / (1.0 + np.exp(log_odds))
        img = color_map(1-p)[..., :3] * 255

        if save_video:
            writer.append_data(img.astype(np.uint8))

    imageio.imwrite(f"{save_dir}/ra_map.png", img.astype(np.uint8))


if __name__ == "__main__":
    cli = tyro.cli(run_sar)
