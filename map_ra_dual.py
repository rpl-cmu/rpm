import os
import tyro
import time
import numpy as np
import imageio
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join as pjoin
from pathlib import Path

import jax
import jax.numpy as jnp

from mmwcas.dataset import ChirpPoseDataset
from sar import RAmapping, MergeDataset
from evaluation import metric
from utils import map_to_pts


import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_sar(
    folder: str,
    name: str = "current_time",
    save_video: bool = False,
    map_extent: float = 10,
    resolution: float = 0.1,
    protect_range: float = 0.2,
    angle_fov: float = 20.0,
    amp_sigma: float = 2.0,
    prob_hit: float = 0.7,
    prob_miss: float = 0.4,
    clamp_log_max: float = 3.5,  # 0.97
    clamp_log_min: float = -2.0,  # 0.12
    angele_fft_size: int = 256,
    f_score_thresh: float = 0.2,
) -> None:

    data_dir = Path(folder)
    seq_name = data_dir.name
    name = str(int(time.time())) if name == "current_time" else name
    save_dir = f"exps/map_ra/{seq_name}_{name}"
    os.makedirs(f"{save_dir}", exist_ok=True)

    lidar_map = pkl.load(open(pjoin(data_dir, "map", "lidar.pkl"), "rb"))
    lidar_pc = map_to_pts(lidar_map["data"], lidar_map["t"], lidar_map["resolution"])

    d0 = ChirpPoseDataset(pjoin(data_dir, "radar0"))
    d1 = ChirpPoseDataset(pjoin(data_dir, "radar1"))
    dataset = MergeDataset(d0, d1)

    mapper = RAmapping(
        dataset=d0,
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
        writer = imageio.get_writer(f"{save_dir}/mapping.mp4", fps=d0.fps * 2)
        color_map = plt.get_cmap("bone")

    for chirps, poses, stamp, sensor in tqdm(dataset):
        pose = poses[0, 4]  # center: chirp 0 tx 4
        log_odds = ramapping(pose, chirps, log_odds)

        p = 1.0 - 1.0 / (1.0 + np.exp(log_odds))
        img = color_map(1 - p)[..., :3] * 255

        if save_video:
            writer.append_data(img.astype(np.uint8))

    imageio.imwrite(f"{save_dir}/prob_map.png", img.astype(np.uint8))
    p = 1.0 - 1.0 / (1.0 + np.exp(log_odds))
    prob, t, res = mapper.map.save_probmap(f"{save_dir}/prob.pkl", p)

    # evaluation
    eval_pc = map_to_pts(prob, t, res)
    cd = metric.chamfer_distance(lidar_pc, eval_pc)
    hd = metric.hausdorff_distance(lidar_pc, eval_pc)
    f_score = metric.f_score(lidar_pc, eval_pc, thresh_dist=f_score_thresh)
    print(f"CD: {cd}, HD: {hd}, F-score: {f_score}")

    fig = plt.figure()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.scatter(lidar_pc[:, 0], lidar_pc[:, 1], c="r", label="Lidar Points", s=1)
    plt.scatter(eval_pc[:, 0], eval_pc[:, 1], c="g", label="Evaluated Points", s=1)
    plt.savefig(f"{save_dir}/eval.png")
    plt.close()


if __name__ == "__main__":
    cli = tyro.cli(run_sar)
