"""Script to run SAR mapping on MIMO radar data."""

import os
import pickle as pkl
import time
import warnings
from os.path import join as pjoin
from typing import Optional

import imageio
import jax
import matplotlib.pyplot as plt
import numpy as np
import tyro
import wandb
from tqdm import tqdm

from rpm.mmwcas.dataset import MIMODataset
from rpm.sar import BackProjection, MergeDataset, OccupancySAR, RadarMap
from rpm.utils import chamfer_distance, f_score, hausdorff_distance, map_to_pts

warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_sar_mapping(
    folder: str,
    name: str = "current_time",
    save_video: bool = False,
    map_extent: float = 10.0,
    protect_range: float = 0.2,
    resolution: float = 0.1,
    sar_fov: float = 90.0,
    occu_fov: float = 15.0,
    smooth_window: int = 1,
    resolution_scale: int = 1,
    amp_sigma: float = 0.07,
    ang_res: float = 0.5,
    prob_hit: float = 0.7,
    prob_miss: float = 0.2,
    clamp_log_max: float = 3.5,  # 0.97
    clamp_log_min: float = -2.0,  # 0.12
    delay_frames: int = 2,
    max_batch: int = 512,
    rx: list[int] = [],
    tx: list[int] = [],
    en_wandb: bool = False,
    f_score_thresh: float = 0.2,
) -> None:
    """Run the SAR mapping process.

    Args:
        folder: Path to the folder containing radar data and map.
        name: Name for the experiment/run.
        save_video: Whether to save a video of the mapping process.
        map_extent: Extent of the map in meters.
        protect_range: Range around the radar to protect from updates.
        resolution: Resolution of the occupancy map in meters.
        sar_fov: Field of view for SAR processing in degrees.
        occu_fov: Angular field of view for occupancy mapping in degrees.
        smooth_window: Size of the smoothing window for back-projection.
        resolution_scale: Scale factor for resolution during processing.
        amp_sigma: Standard deviation for amplitude-based occupancy updates.
        ang_res: Angular resolution for occupancy mapping in degrees.
        prob_hit: Probability of hit for occupancy mapping.
        prob_miss: Probability of miss for occupancy mapping.
        clamp_log_max: Maximum log-odds value to clamp to.
        clamp_log_min: Minimum log-odds value to clamp to.
        delay_frames: Number of frames to delay before updating occupancy map.
        max_batch: Maximum batch size for processing poses.
        rx: List of receiver indices to use.
        tx: List of transmitter indices to use.
        en_wandb: Whether to enable Weights & Biases logging.
        f_score_thresh: Threshold distance for F-score calculation.
    """
    if en_wandb:
        wandb.init(project="mm_map", config=locals())
        name = wandb.run.name  # type: ignore

    data_dir = folder
    seq_name = data_dir.split("/")[-1]
    name = str(int(time.time())) if name == "current_time" else name
    save_dir = f"exps/map_sar/{seq_name}_{name}"
    os.makedirs(f"{save_dir}", exist_ok=True)

    data0 = MIMODataset(pjoin(data_dir, "radar0"), rx=rx, tx=tx)
    data1 = MIMODataset(pjoin(data_dir, "radar1"), rx=rx, tx=tx)
    dataset = MergeDataset(data0, data1)

    ref_data = data0
    lidar_map = pkl.load(open(pjoin(data_dir, "map", "lidar.pkl"), "rb"))
    lidar_pc = map_to_pts(lidar_map["data"], lidar_map["t"], lidar_map["resolution"])

    print(
        f"synthetic antennas: {len(dataset)} x {ref_data.adc.param.numChirp} x {len(ref_data.rx)} x {len(ref_data.tx)}"
    )
    n_sig = ref_data.adc.param.numChirp  # * len(dataset.rx) * len(dataset.tx)

    back_projection = BackProjection(
        ref_data,
        map_extent,
        protect_range,
        sar_fov,
        smooth_window,
        resolution_scale,
        resolution,
        amp_sigma,
    )
    occuMap = OccupancySAR(
        back_projection.map,
        n_sig,
        amp_sigma,
        back_projection.proc_range_res,
        protect_range,
        map_extent,
        ang_fov=occu_fov,
        ang_res=ang_res,
        prob_hit=prob_hit,
        prob_miss=prob_miss,
        clamp_log_max=clamp_log_max,
        clamp_log_min=clamp_log_min,
    )
    log_map = occuMap.log_map
    prob_mapping = jax.jit(occuMap.__call__)
    color_map = plt.get_cmap("bone")
    color_map_hot = plt.get_cmap("hot")
    writer = None

    if save_video:
        writer = imageio.get_writer(f"{save_dir}/mapping.mp4", fps=ref_data.fps * 2)

    map_state = back_projection.map.get_map()
    num_samples = ref_data.adc.param.numADCSample

    pose_que = []
    sig, pose_tx, pose_rx, _, _ = dataset[0]
    pose_tx = pose_tx.reshape(-1, 4, 4)
    l = pose_tx.shape[0]
    if l > max_batch:
        n_batch = l // max_batch
        batch = np.array_split(np.arange(n_batch * max_batch), n_batch)
        if l % max_batch != 0:
            batch.append(np.arange(l)[-(l % max_batch) :])
    else:
        batch = [np.arange(l)]

    for sig, pose_tx, pose_rx, stamp, sensor in tqdm(dataset):
        sig = sig.reshape(-1, num_samples)
        pose_tx = pose_tx.reshape(-1, 4, 4)
        pose_rx = pose_rx.reshape(-1, 4, 4)

        for b in batch:
            map_state = back_projection.update_batch(
                pose_tx[b], pose_rx[b], sig[b], map_state
            )
            pose_que.append(pose_tx[b[0]])

        if len(pose_que) > len(batch) * delay_frames:
            for i in range(len(batch)):
                log_map = prob_mapping(pose_que.pop(0), map_state, log_map)

        if writer is not None:
            prob = 1.0 - 1.0 / (1.0 + np.exp(log_map))
            img = color_map(1 - prob)[:, :, :3] * 255

            map_abs = np.abs(map_state["complex"])
            map_abs = map_abs / map_state["n_obs"]
            map_abs *= map_state["n_obs"] > n_sig
            prob = 1 - np.exp(-(map_abs**2) / (2 * amp_sigma**2))
            map = color_map_hot(prob)[:, :, :3] * 255

            img = np.concat((img, map), axis=1)
            writer.append_data(img.astype(np.uint8))

    prob = 1.0 - 1.0 / (1.0 + np.exp(log_map))
    img = color_map(1 - prob)[:, :, :3] * 255
    imageio.imwrite(f"{save_dir}/prob_map.png", img.astype(np.uint8))
    back_projection.map.update_map(map_state)
    back_projection.map.visualize(save_dir, color_map="hot")
    data, t, res = back_projection.map.save_probmap(f"{save_dir}/prob.pkl", prob)
    back_projection.map.save(save_dir)

    # evaluation
    eval_pc = map_to_pts(data, t, res)
    cd = chamfer_distance(lidar_pc, eval_pc)
    hd = hausdorff_distance(lidar_pc, eval_pc)
    fs = f_score(lidar_pc, eval_pc, thresh_dist=f_score_thresh)
    print(f"CD, HD, F-score\n{cd}, {hd}, {fs}")
    with open(f"{save_dir}/eval.txt", "w") as f:
        f.write(f"CD, HD, F-score\n{cd}, {hd}, {fs}")

    fig = plt.figure()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.scatter(lidar_pc[:, 0], lidar_pc[:, 1], c="r", label="Lidar Points", s=1)
    plt.scatter(eval_pc[:, 0], eval_pc[:, 1], c="g", label="Evaluated Points", s=1)
    plt.savefig(f"{save_dir}/eval.png")
    plt.close()

    if en_wandb:
        log_files = os.listdir(save_dir)
        for file in log_files:
            if file.endswith(".png"):
                wandb.log({file.split(".")[0]: wandb.Image(pjoin(save_dir, file))})
        wandb.log({"CD": cd, "HD": hd, "F-score": f_score})
        wandb.finish()


if __name__ == "__main__":
    cli = tyro.cli(run_sar_mapping)
