import os
import cv2
import time
import numpy as np
import argparse
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join as pjoin

import jax
from jaxtyping import PyTree, Complex, Float, Int, Array

from sar import MIMODataset, BackProjection, RadarMap


class Runner:
    def __init__(self, args, r="radar0"):

        # load existing map
        if args.load:
            map_sar = RadarMap.load(args.load)
            map_sar.visualize(args.load, color_map="hot")
            return

        data_dir = args.folder
        seq_name = data_dir.split("/")[-1]
        name = int(time.time()) if args.name is None else args.name
        save_dir = f"exps/{seq_name}_{name}"
        os.makedirs(f"{save_dir}", exist_ok=True)

        self.dataset = MIMODataset(
            pjoin(data_dir, r),
            # rx=[0],
            tx=[0],
            rx=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
        )

        self.back_projection = BackProjection(
            self.dataset,
            args.extent,
            protect_range=args.protect_range,
            azimuth_fov=args.fov,
        )

        if args.video:
            self.writer = imageio.get_writer(
                f"{save_dir}/mapping.mp4", fps=self.dataset.fps
            )
            self.color_map = plt.get_cmap("hot")

        map_state = self.run_sar(args.max_batch, log_video=args.video)
        self.back_projection.map.update_map(map_state)
        self.back_projection.map.visualize(save_dir, color_map="hot")
        self.back_projection.map.save(save_dir)

    def run_sar(self, max_batch: int, log_video: bool = False) -> dict:

        map_state = self.back_projection.map.get_map()
        num_samples = self.dataset.adc.param.numADCSample

        sig, pose_tx, pose_rx = self.dataset[0]
        pose_tx = pose_tx.reshape(-1, 4, 4)
        l = pose_tx.shape[0]
        if l > max_batch:
            n_batch = l // max_batch
            batch = np.array_split(np.arange(n_batch * max_batch), n_batch)
            if l % max_batch != 0:
                batch.append(np.arange(l)[-(l % max_batch) :])
        else:
            batch = [np.arange(l)]

        for sig, pose_tx, pose_rx in tqdm(self.dataset):
            sig = sig.reshape(-1, num_samples)
            pose_tx = pose_tx.reshape(-1, 4, 4)
            pose_rx = pose_rx.reshape(-1, 4, 4)

            # map_state["complex"] = np.zeros_like(map_state["complex"])
            for b in batch:
                map_state = self.back_projection.update_batch(
                    pose_tx[b], pose_rx[b], sig[b], map_state
                )

            if log_video:
                map_abs = np.abs(map_state["complex"])
                left, right = np.percentile(map_abs, np.array([0.0, 99.0]))
                map_clip = (np.clip(map_abs, left, right) - left) / (right - left)
                map_clip = map_clip / np.max(map_clip)
                map = self.color_map(map_clip)[:, :, :3] * 255
                self.writer.append_data(map.astype(np.uint8))
                # map_update = map_state['update'].astype(np.uint8)*255
                # self.writer.append_data(np.asarray(map_update))

        return map_state


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Create SAR map")
    argparser.add_argument("--folder", type=str, help="Folder path", required=True)
    argparser.add_argument("--load", type=str, help="Load exp", default=None)

    argparser.add_argument("--video", type=bool, help="save video", default=False)
    argparser.add_argument("--max_batch", type=int, help="batch size", default=1024)
    argparser.add_argument(
        "--extent",
        type=float,
        help="map area extent from pose position",
        default=10,  # ms
    )
    argparser.add_argument(
        "--protect_range", type=float, help="protect range", default=0.6
    )
    argparser.add_argument(
        "--fov", type=float, help="half azimuth field of view", default=20
    )
    argparser.add_argument("--name", type=str, help="name of the case", default=None)

    args = argparser.parse_args()

    runner = Runner(args)
