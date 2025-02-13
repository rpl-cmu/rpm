import os
import time
import numpy as np
import argparse
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join as pjoin
from functools import partial

import jax
import optax
import jax.numpy as jnp
from jaxtyping import PyTree, Complex, Float, Int, Array

from mmwcas.dataset import MIMODataset
from sar import AFBackProjection, RadarMap


class Runner:
    def __init__(self, args, r="radar0"):

        # load existing map
        if args.load:
            map_state = RadarMap.load(args.load)
            RadarMap.visualize(map_state, args.load, color_map="hot")
            return

        data_dir = args.folder
        seq_name = data_dir.split("/")[-1]
        name = int(time.time()) if args.name is None else args.name
        save_dir = f"exps/{seq_name}_{name}"
        os.makedirs(f"{save_dir}", exist_ok=True)

        self.dataset = MIMODataset(
            pjoin(data_dir, r),
            rx=[0],
            tx=[0],
        )

        self.back_projection = AFBackProjection(
            # back_projection = BackProjection(
            self.dataset,
            args.extent,
            protect_range=args.protect_range,
            azimuth_fov=args.fov,
        )

        self.opt = optax.adam(learning_rate=1e-3)

        if args.video:
            self.writer = imageio.get_writer(
                f"{save_dir}/mapping.mp4", fps=self.dataset.fps
            )
            self.color_map = plt.get_cmap("hot")

        map_state = self.run_sar(args.max_batch, args.af_step, log_video=args.video)
        RadarMap.visualize(map_state, save_dir, color_map="hot")
        RadarMap.save(save_dir, map_state)

    def back_project(self, pose_tx, t_rx, sig, map_state, t_rx_init):
        map_state = self.back_projection.batch_project(pose_tx, t_rx, sig, map_state)

        mask = map_state["update"]
        map_abs = jnp.abs(map_state["complex"]) * mask
        map_abs = jnp.clip(map_abs, 1e-10, 1e10)
        norm_abs = map_abs / jnp.sum(map_abs)
        entropy = -jnp.sum(norm_abs * jnp.log(norm_abs))
        t_rx_diff = jnp.abs(t_rx - t_rx_init).sum()
        loss = entropy + t_rx_diff

        map_state["entropy"] = entropy
        map_state["t_rx_diff"] = t_rx_diff

        return loss, map_state

    def _step(self, posetx, t_rx, sig, map_state, t_rx_init, opt_state):

        (loss, map_new), grad = jax.value_and_grad(
            self.back_project, argnums=1, has_aux=True
        )(posetx, t_rx, sig, map_state, t_rx_init)
        updates, opt_state = self.opt.update(grad, opt_state)
        t_rx = optax.apply_updates(t_rx, updates)

        return loss, map_new, t_rx, opt_state

    def run_sar(self, max_batch: int, af_step:int, log_video: bool = False) -> dict:

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

        bp = jax.jit(self._step)
        frame = 0
        for sig, pose_tx, pose_rx in tqdm(self.dataset):
            sig = sig.reshape(-1, num_samples)
            pose_tx = pose_tx.reshape(-1, 4, 4)
            pose_rx = pose_rx.reshape(-1, 4, 4)

            # map_state["complex"] = np.zeros_like(map_state["complex"])
            for b in batch:
                nstep = af_step if frame > 20 else 1

                t_rx = pose_rx[b][:, :3, 3]
                opt_state = self.opt.init(t_rx)
                t_rx_init = jnp.copy(t_rx)

                for i in range(nstep):
                    loss, map_new, t_rx, opt_state = bp(
                        pose_tx[b], t_rx, sig[b], map_state, t_rx_init, opt_state
                    )
                    # print(
                    #     f'loss: {loss}, entropy: {map_new["entropy"]}, pos_diff: {map_new["t_rx_diff"]}'
                    # )

                map_state = map_new

            frame += 1
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
    argparser.add_argument("--max_batch", type=int, help="batch size", default=256)
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
    argparser.add_argument("--af_step", type=int, help="autofocus step", default=20)

    args = argparser.parse_args()

    runner = Runner(args)
