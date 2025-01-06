import os
import tyro
import time
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join as pjoin
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Complex, Float, Array

from mmwcas.dataset import MIMODataset
from sar import BackProjection, RadarMap

from mmwcas.process import PatternAZI

pattern = PatternAZI(xla=jnp)


def project_pix(
    pose_tx: Float[Array, "4 4"],
    pose_rx: Float[Array, "4 4"],
    sig: Complex[Array, "n_samples"],
    pixel: Float[Array, "3"],
    k: Complex[Array, " "],
    window: Float[Array, "n_samples"],
    range_res: Float,
    range_max: Float,
    protect_range: Float,
    azimuth_fov: Float,
) -> Complex[Array, ""]:
    pts = pose_tx[:3, 3]
    ray_v = jnp.dot(pose_tx[:3, :3], jnp.array([1, 0, 0]))
    ray = pixel - pts
    rtx = jnp.linalg.norm(pixel - pose_tx[:3, 3])
    rrx = jnp.linalg.norm(pixel - pose_rx[:3, 3])
    r = (rtx + rrx) * 0.5

    ray_dot = ray @ ray_v / r
    angle = jnp.rad2deg(jnp.arccos(ray_dot))

    mask_range = jnp.logical_and(r < range_max, r > protect_range)
    mask_angle = jnp.logical_and(angle > 0, angle < azimuth_fov)
    mask = jnp.logical_and(mask_range, mask_angle)

    sig_fft = jnp.fft.fft(window * sig)
    idx = (r // range_res).astype(int)
    sig_min, sig_max = sig_fft[idx], sig_fft[idx + 1]
    sig_at_r = sig_min + (r % range_res) * (sig_max - sig_min) / range_res
    image = sig_at_r * jnp.exp(-1j * 2 * k * r)
    gain = jnp.power(10, pattern.gain(angle) / 10)
    image = image * gain

    image = image * mask

    return image


def run_sar(
    folder: str,
    load: str,
    map_extent: float = 10.0,
    protect_range: float = 0.6,
    azimuth_fov: float = 20.0,  # degree
    max_batch: int = 1024,
    r: str = "radar0",
    rx: list[int] = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    tx: list[int] = [0],
) -> None:

    data_dir = folder
    map_sar = RadarMap.load(load)
    save_dir = f"{load}/pix_analysis"
    colormap = plt.get_cmap("hot")
    os.makedirs(save_dir, exist_ok=True)
    dataset = MIMODataset(pjoin(data_dir, r), rx=rx, tx=tx)
    C = dataset.adc.param.speedOfLight
    F0 = dataset.adc.param.startFreqConst
    k = 2 * np.pi * (F0 / C)
    window = jnp.hanning(dataset.adc.param.numADCSample)
    range_res = dataset.adc.param.rangeResolution
    range_max = dataset.adc.param.maxRange

    map_abs = np.abs(map_sar.complex)
    thresh = np.percentile(map_abs, 99.0)
    map_img = np.clip(map_abs / thresh, 0, 1)
    map_img = colormap(map_img)[:, :, :3] * 255
    map_img = map_img[:, :, ::-1]
    map_img = map_img.astype(np.uint8)

    def click_event(event, c, r, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Pixel Value at ({c}, {r}): {map_sar.complex[r, c]}")
            print(f"abs: {np.clip(np.abs(map_sar.complex[r,c]) / thresh, 0, 1)}")
            pix_vals = []
            target = map_sar.grid[r, c]
            print(target)
            map_copy = map_img.copy()
            cv2.circle(map_copy, (c, r), 5, (255, 0, 0), -1)
            cv2.imwrite(f"{save_dir}/map_pix{r}_{c}.png", map_copy)

            project = partial(
                project_pix,
                pixel=target,
                k=k,
                window=window,
                range_res=range_res,
                range_max=range_max,
                protect_range=protect_range,
                azimuth_fov=azimuth_fov,
            )
            project_batch = jax.jit(jax.vmap(project))

            for sig, pose_tx, pose_rx, stamp in tqdm(dataset):

                sig = sig.reshape(-1, dataset.adc.param.numADCSample)
                pose_tx = pose_tx.reshape(-1, 4, 4)
                pose_rx = pose_rx.reshape(-1, 4, 4)
                dis = np.linalg.norm(target - pose_tx[:, :3, 3], axis=1)
                mask = np.where(dis < range_max)[0]

                if len(mask) == 0:
                    continue

                vals = project_batch(pose_tx[mask], pose_rx[mask], sig[mask])
                vals = np.asarray(vals[np.abs(vals) > 0])
                if len(vals) > 0:
                    pix_vals.append(vals)

            pix_vals = np.concatenate(pix_vals)
            np.save(f"{save_dir}/pix_vals{r}_{c}.npy", pix_vals)
            final_val = np.sum(pix_vals)
            print(final_val, np.clip(np.abs(final_val) / thresh, 0, 1))
            # plt.scatter(np.real(pix_vals), np.imag(pix_vals), s=0.5)
            # plt.scatter(np.real(final_val), np.imag(final_val), c="r")
            # plt.savefig(f"{save_dir}/pix_vals{r}_{c}.png", bbox_inches="tight")
            # plt.show()

    cv2.imshow("Image", map_img)
    cv2.setMouseCallback("Image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cli = tyro.cli(run_sar)
