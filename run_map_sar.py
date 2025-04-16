import os
import cv2
import tyro
import wandb
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join as pjoin

import jax
from mmwcas.dataset import MIMODataset
from sar import BackProjection, RadarMap, OccupancySAR


def run_sar(
    folder: str,
    load: str = None,
    name: str = "current_time",
    save_video: bool = False,
    map_extent: float = 10.0,
    protect_range: float = 0.3,
    resolution: float = 0.1,
    azimuth_fov: float = 20.0,
    bp_extra_fov: float = 10.0,
    smooth_window: int = 1,
    resolution_scale: int = 1,
    amp_sigma: float = 0.15,
    ang_res: float = 0.5,
    prob_hit: float = 0.99,
    prob_miss: float = 0.4,
    clamp_log_max: float = 3.5,  # 0.97
    clamp_log_min: float = -2.0,  # 0.12
    max_batch: int = 512,
    r: str = "radar0",
    rx: list[int] = [],
    tx: list[int] = [],
    en_wandb: bool = False,
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
    print(folder)
    # load existing map
    if load:
        map_sar = RadarMap.load(load)
        map_sar.visualize(load, color_map="hot")
        return

    if en_wandb:
        wandb.init(project="mm_map", config=locals())
        name = wandb.run.name

    data_dir = folder
    seq_name = data_dir.split("/")[-1]
    name = str(int(time.time())) if name == "current_time" else name
    save_dir = f"exps/map_sar/{seq_name}_{name}"
    os.makedirs(f"{save_dir}", exist_ok=True)

    dataset = MIMODataset(pjoin(data_dir, r), rx=rx, tx=tx)

    print(
        f"synthetic antennas: {len(dataset)} x {dataset.adc.param.numChirp} x {len(dataset.rx)} x {len(dataset.tx)}"
    )
    n_sig = dataset.adc.param.numChirp  # * len(dataset.rx) * len(dataset.tx)

    back_projection = BackProjection(
        dataset,
        map_extent,
        protect_range,
        azimuth_fov + bp_extra_fov,
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
        angle_fov=azimuth_fov,
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
        writer = imageio.get_writer(f"{save_dir}/mapping.mp4", fps=dataset.fps)

    map_state = back_projection.map.get_map()
    num_samples = dataset.adc.param.numADCSample

    sig, pose_tx, pose_rx, stamp = dataset[0]
    pose_tx = pose_tx.reshape(-1, 4, 4)
    l = pose_tx.shape[0]
    if l > max_batch:
        n_batch = l // max_batch
        batch = np.array_split(np.arange(n_batch * max_batch), n_batch)
        if l % max_batch != 0:
            batch.append(np.arange(l)[-(l % max_batch) :])
    else:
        batch = [np.arange(l)]

    for sig, pose_tx, pose_rx, stamp in tqdm(dataset):

        sig = sig.reshape(-1, num_samples)
        pose_tx = pose_tx.reshape(-1, 4, 4)
        pose_rx = pose_rx.reshape(-1, 4, 4)

        for b in batch:
            map_state = back_projection.update_batch(
                pose_tx[b], pose_rx[b], sig[b], map_state
            )

        log_map = prob_mapping(pose_tx[0], map_state, log_map)

        if save_video:
            prob = 1.0 - 1.0 / (1.0 + np.exp(log_map))
            img = color_map(1 - prob)[:, :, :3] * 255
            writer.append_data(img.astype(np.uint8))

    prob = 1.0 - 1.0 / (1.0 + np.exp(log_map))
    img = color_map(1 - prob)[:, :, :3] * 255
    cv2.imwrite(f"{save_dir}/prob_map.png", img.astype(np.uint8)[..., ::-1])
    back_projection.map.update_map(map_state)
    back_projection.map.visualize(save_dir, color_map="hot")
    back_projection.map.save_probmap(f"{save_dir}/prob.pkl", prob)
    # back_projection.map.save(save_dir)

    if en_wandb:
        log_files = os.listdir(save_dir)
        for file in log_files:
            if file.endswith(".png"):
                wandb.log({file.split(".")[0]: wandb.Image(pjoin(save_dir, file))})
        wandb.finish()


if __name__ == "__main__":
    cli = tyro.cli(run_sar)
