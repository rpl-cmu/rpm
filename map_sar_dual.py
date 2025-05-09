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
from os.path import join as pjoin

import jax
from mmwcas.dataset import MIMODataset
from sar import BackProjection, RadarMap, OccupancySAR, MergeDataset
from evaluation import metric
from utils import map_to_pts

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_sar(
    folder: str,
    load: str = None,
    name: str = "current_time",
    save_video: bool = False,
    map_extent: float = 10.0,
    protect_range: float = 0.2,
    resolution: float = 0.1,
    sar_fov: float = 90.0,
    occu_fov: float = 20.0,
    smooth_window: int = 1,
    resolution_scale: int = 1,
    amp_sigma: float = 0.1,
    ang_res: float = 0.25,
    prob_hit: float = 1,
    prob_miss: float = 0.46,
    clamp_log_max: float = 3.5,  # 0.97
    clamp_log_min: float = -2.0,  # 0.12
    max_batch: int = 512,
    rx: list[int] = [],
    tx: list[int] = [],
    en_wandb: bool = False,
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
        writer = imageio.get_writer(f"{save_dir}/mapping.mp4", fps=ref_data.fps * 2)

    map_state = back_projection.map.get_map()
    num_samples = ref_data.adc.param.numADCSample

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
            log_map = prob_mapping(pose_tx[b[0]], map_state, log_map)

        if save_video:
            prob = 1.0 - 1.0 / (1.0 + np.exp(log_map))
            img = color_map(1 - prob)[:, :, :3] * 255
            writer.append_data(img.astype(np.uint8))

            # map_abs = np.abs(map_state["complex"])
            # map_abs = map_abs / map_state["n_obs"]
            # map_abs *= map_state["n_obs"] > n_sig
            # prob = 1 - np.exp(-(map_abs**2) / (2 * amp_sigma**2))
            # map = color_map(prob)[:, :, :3] * 255
            # writer.append_data(map.astype(np.uint8))

    prob = 1.0 - 1.0 / (1.0 + np.exp(log_map))
    img = color_map(1 - prob)[:, :, :3] * 255
    cv2.imwrite(f"{save_dir}/prob_map.png", img.astype(np.uint8)[..., ::-1])
    back_projection.map.update_map(map_state)
    back_projection.map.visualize(save_dir, color_map="hot")
    data, t, res = back_projection.map.save_probmap(f"{save_dir}/prob.pkl", prob)

    # evaluation
    eval_pc = map_to_pts(data, t, res)
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

    if en_wandb:
        log_files = os.listdir(save_dir)
        for file in log_files:
            if file.endswith(".png"):
                wandb.log({file.split(".")[0]: wandb.Image(pjoin(save_dir, file))})
        wandb.log({"CD": cd, "HD": hd, "F-score": f_score})
        wandb.finish()


if __name__ == "__main__":
    cli = tyro.cli(run_sar)
