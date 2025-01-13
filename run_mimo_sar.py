import os
import tyro
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join as pjoin

from mmwcas.dataset import MIMODataset
from sar import BackProjection, RadarMap


def run_sar(
    folder: str,
    load: str = None,
    name: str = "current_time",
    save_video: bool = False,
    map_extent: float = 10.0,
    protect_range: float = 0.4,
    azimuth_fov: float =20.0,  # degree
    smooth_window: int = 5,
    max_batch: int = 1024,
    r: str = "radar0",
    rx: list[int] = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
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
        rx: receive antennas idx (None = all antennas)
        tx: transmit antennas idx (None = all antennas)
    """
    if len(rx) == 0:
        rx = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    if len(tx) == 0:
        tx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # load existing map
    if load:
        map_sar = RadarMap.load(load)
        map_sar.visualize(load, color_map="hot")
        return

    data_dir = folder
    seq_name = data_dir.split("/")[-1]
    name = str(int(time.time())) if name == "current_time" else name
    save_dir = f"exps/{seq_name}_{name}"
    os.makedirs(f"{save_dir}", exist_ok=True)

    dataset = MIMODataset(pjoin(data_dir, r), rx=rx, tx=tx)

    print(
        "synthetic antennas: ",
        len(dataset) * dataset.adc.param.numChirp * len(tx) * len(rx),
    )

    back_projection = BackProjection(
        dataset, map_extent, protect_range, azimuth_fov, smooth_window
    )

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

        # map_state["complex"] = np.zeros_like(map_state["complex"])
        for b in batch:
            map_state = back_projection.update_batch(
                pose_tx[b], pose_rx[b], sig[b], map_state
            )

        if save_video:
            map_abs = np.abs(map_state["complex"])
            left, right = np.percentile(map_abs, np.array([0.0, 99.0]))
            map_clip = (np.clip(map_abs, left, right) - left) / (right - left)
            map_clip = map_clip / np.max(map_clip)
            map = color_map(map_clip)[:, :, :3] * 255
            writer.append_data(map.astype(np.uint8))
            # map_update = map_state['update'].astype(np.uint8)*255
            # writer.append_data(np.asarray(map_update))

    back_projection.map.update_map(map_state)
    back_projection.map.visualize(save_dir, color_map="hot")
    back_projection.map.save(save_dir)


if __name__ == "__main__":
    cli = tyro.cli(run_sar)
