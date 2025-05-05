import os
import cv2
import tyro
import wandb
import time
import imageio
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from os.path import join as pjoin

import jax
from mmwcas.dataset import MIMODataset
from sar import BackProjection, RadarMap, OccupancySAR
from evaluation import metric
from utils import map_to_pts


def run_sar(
    folder: str,
    load: str,
    save_video: bool = False,
    map_extent: float = 10.0,
    protect_range: float = 0.2,
    resolution: float = 0.1,
    occu_fov: float = 90.0,
    smooth_window: int = 1,
    resolution_scale: int = 1,
    amp_sigma: float = 0.05,
    ang_res: float = 0.25,
    prob_hit: float = 0.9,
    prob_miss: float = 0.46,
    clamp_log_max: float = 3.5,  # 0.97
    clamp_log_min: float = -2.0,  # 0.12
    max_batch: int = 512,
    r: str = "radar0",
    rx: list[int] = [],
    tx: list[int] = [],
    f_score_thresh: float = 0.2,
) -> None:
    """Run the SAR mapping process

    Args:
        folder: path to the dataset
        name: name of the output folder
        save_video: whether to save the video
        map_extent: extent of the map in meters
        protect_range: range to protect in meters
        azimuth_fov: azimuth field of view in degrees
        bp_extra_fov: extra field of view for back projection in degrees
        smooth_window: smoothing window size
        resolution_scale: scale fft resolution
        resolution: map resolution in meters
        amp_sigma: amplitude sigma for the probability map
        ang_res: angular resolution in degrees for prob mapping
        prob_hit: probability of hit
        prob_miss: probability of miss
        clamp_log_max: maximum value for the log map
        clamp_log_min: minimum value for the log map
        max_batch: maximum batch size
        r: radar name
        rx: list of receive antennas (leave empty to use all)
        tx: list of transmit antennas (leave empty to use all)
        en_wandb: whether to enable wandb logging (will override name)
    """

    map_sar = RadarMap.load(load)

    data_dir = folder
    save_dir = load

    dataset = MIMODataset(pjoin(data_dir, r), rx=rx, tx=tx)
    lidar_map = pkl.load(open(pjoin(data_dir, "map", "lidar.pkl"), "rb"))
    lidar_pc = map_to_pts(lidar_map["data"], lidar_map["t"], lidar_map["resolution"])

    n_sig = dataset.adc.param.numChirp  # * len(dataset.rx) * len(dataset.tx)

    occuMap = OccupancySAR(
        map_sar,
        n_sig,
        amp_sigma,
        resolution,
        protect_range,
        map_extent,
        angle_fov=occu_fov,
        ang_res=ang_res,
        prob_hit=prob_hit,
        prob_miss=prob_miss,
        clamp_log_max=clamp_log_max,
        clamp_log_min=clamp_log_min,
    )
    log_map = occuMap.log_map
    prob_mapping = jax.jit(occuMap.__call__)
    color_map = plt.get_cmap("bone")

    if save_video:
        writer = imageio.get_writer(f"{save_dir}/trace_prob.mp4", fps=dataset.fps)

    map_state = map_sar.get_map()

    for sig, pose_tx, pose_rx, stamp in tqdm(dataset):

        pose_tx = pose_tx.reshape(-1, 4, 4)
        log_map = prob_mapping(pose_tx[0], map_state, log_map)

        if save_video:
            prob = 1.0 - 1.0 / (1.0 + np.exp(log_map))
            img = color_map(1 - prob)[:, :, :3] * 255
            img = img.astype(np.uint8)
            writer.append_data(img)

    prob = 1.0 - 1.0 / (1.0 + np.exp(log_map))
    img = color_map(1 - prob)[:, :, :3] * 255
    cv2.imwrite(f"{save_dir}/prob_map.png", img.astype(np.uint8)[..., ::-1])
    data, t, res = map_sar.save_probmap(f"{save_dir}/prob.pkl", prob)

    # evaluation
    eval_pc = map_to_pts(data, t, res)
    cd = metric.chamfer_distance(lidar_pc, eval_pc)
    hd = metric.hausdorff_distance(lidar_pc, eval_pc)
    f_score = metric.f_score(lidar_pc, eval_pc, thresh_dist=f_score_thresh)
    print(f"CD: {cd}, HD: {hd}, F-score: {f_score}")


if __name__ == "__main__":
    cli = tyro.cli(run_sar)
