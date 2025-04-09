import os
import cv2
import tyro
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
    protect_range: float = 0.4,
    azimuth_fov: float = 20.0,  # degree
    smooth_window: int = 1,
    resolution_scale: int = 1,
    amp_sigma: float = 0.2,
    max_batch: int = 512,
    r: str = "radar0",
    rx: list[int] = [0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15],
    tx: list[int] = [0],
) -> None:
    """Run the SAR mapping process

    Args:
        folder: path to the dataset
        name: name of the experiment
        load: path to the existing map folder
        save_video: save the mapping process as a video
        map_extent: distance to extent from pose position to create the map
        protect_range: minimum range to process from pose position
        azimuth_fov: field of view of the azimuth angle (degree)
        smooth_window: window size for freqeuncy spectrum peak suppression
        max_batch: batch size for the mapping process
        r: radar name
        rx: receive antennas idx (leave empty if want to use all antennas)
        tx: transmit antennas idx (leave empty if want to use all antennas)
    """

    # load existing map
    if load:
        map_sar = RadarMap.load(load)
        map_sar.visualize(load, color_map="hot")
        return

    data_dir = folder
    seq_name = data_dir.split("/")[-1]
    name = str(int(time.time())) if name == "current_time" else name
    save_dir = f"occusar/{seq_name}_{name}"
    os.makedirs(f"{save_dir}", exist_ok=True)

    dataset = MIMODataset(pjoin(data_dir, r), rx=rx, tx=tx)

    print(
        f"synthetic antennas: {len(dataset)} x {dataset.adc.param.numChirp} x {len(dataset.rx)} x {len(dataset.tx)}"
    )
    n_sig = dataset.adc.param.numChirp * len(dataset.rx) * len(dataset.tx)

    back_projection = BackProjection(
        dataset,
        map_extent,
        protect_range,
        azimuth_fov,
        smooth_window,
        resolution_scale,
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
    )
    log_map = occuMap.log_map
    prob_mapping = jax.jit(occuMap.__call__)

    if save_video:
        writer = imageio.get_writer(f"{save_dir}/mapping.mp4", fps=dataset.fps)
        color_map = plt.get_cmap("hot")

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
            img = color_map(prob)[:, :, :3] * 255
            writer.append_data(img.astype(np.uint8))

    prob = 1.0 - 1.0 / (1.0 + np.exp(log_map))
    img = color_map(prob)[:, :, :3] * 255
    cv2.imwrite(f"{save_dir}/prob_map.png", img.astype(np.uint8)[..., ::-1])
    back_projection.map.update_map(map_state)
    back_projection.map.visualize(save_dir, color_map="hot")
    back_projection.map.save(save_dir)


if __name__ == "__main__":
    cli = tyro.cli(run_sar)
