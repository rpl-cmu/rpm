import jax.scipy.optimize
import numpy as np
from functools import cached_property, partial

import jax
import jax.numpy as jnp
from jax.scipy.stats import vonmises
from jaxtyping import PyTree, Complex, Float, Array

from sar import RadarMap
from mmwcas.dataset import ChirpPoseDataset
from mmwcas.process import PatternAZI


class BackProjectionProb:
    def __init__(
        self,
        dataset: ChirpPoseDataset,
        map_extent: float = 10,  # meter
        protect_range: float = 0.5,  # meter
        azimuth_fov: float = 20,  # degree
        smooth_window: int = 5,
        resolution_scale: int = 2,
    ) -> None:
        # frame poses
        poses = dataset.chirp_poses[:, 0, 0, :, :]

        # signal parameters
        param = dataset.adc.param
        self.range_res = param.rangeResolution
        self.range_max = param.maxRange
        C = param.speedOfLight
        F0 = param.startFreqConst
        u = param.chirpSlope
        self.k = 2 * np.pi * (F0 / C)
        self.window = jnp.hanning(param.numADCSample)

        self.smooth_kernel = jnp.ones(smooth_window) / smooth_window

        # radiation pattern parameters
        self.pattern = PatternAZI(xla=jnp)
        self.azimuth_fov = azimuth_fov

        # map parameters
        self.protect_range = protect_range
        self.proc_range_res = param.rangeResolution / resolution_scale
        self.fft_len = param.numADCSample * resolution_scale
        self.map = RadarMap(
            resolution=self.proc_range_res,
            poses=poses,
            map_extent=map_extent,
        )

        self.sigma_amp = 10.0
        self.k_phase = 4
        self.phase_level = jnp.linspace(-jnp.pi, jnp.pi, 18)
        self.xi = -1e-3

        self.map_logodds = jnp.zeros(self.map.grid.shape[:2] + (len(self.phase_level),))

    def adjust_loss(self, zeta, prob):
        p = jnp.clip(prob + zeta, 0, 1)
        log_odds = jnp.log(p / (1 - p + 1e-7))
        logsum = jnp.sum(log_odds, axis=-1, keepdims=True)
        loss = jnp.linalg.norm(logsum)
        return loss

    def project_sig2D(
        self,
        pose_tx: Float[Array, "4 4"],
        pose_rx: Float[Array, "4 4"],
        sig: Complex[Array, "n_samples"],
        grid: Float[Array, "n m 3"],
        k: Complex[Array, " "],
    ) -> Float[Array, "n m l"]:

        n, m = grid.shape[:2]
        l = len(self.phase_level)

        pts = pose_tx[:3, 3]
        pixels = grid.reshape(-1, 3)
        ray_v = jnp.dot(pose_tx[:3, :3], jnp.array([1, 0, 0]))

        rtx = jnp.linalg.norm(pixels - pose_tx[:3, 3], axis=-1)
        rrx = jnp.linalg.norm(pixels - pose_rx[:3, 3], axis=-1)
        r = (rtx + rrx) * 0.5
        sig = sig - jnp.mean(sig)
        sig_fft = jnp.fft.fft(self.window * sig, n=self.fft_len, norm="forward")
        # mag = jnp.abs(sig_fft)
        # mag_smooth = jnp.convolve(mag, self.smooth_kernel, mode="same")
        # sig_fft = sig_fft / mag * mag_smooth

        # a = jnp.abs(sig_fft)
        # sigma = jnp.sqrt(0.5*jnp.mean(a**2))
        # print(a.shape, sigma.shape, sigma)

        idx = (r // self.proc_range_res).astype(int)
        sig_l, sig_r = sig_fft[idx], sig_fft[idx + 1]
        sig_at_r = (
            sig_l + (r % self.proc_range_res) * (sig_r - sig_l) / self.proc_range_res
        )

        ray_dot = (pixels - pts) @ ray_v / r
        angle = jnp.rad2deg(jnp.arccos(ray_dot))
        # gain = jnp.power(10, self.pattern.gain(angle) / 10)

        mask_range = jnp.logical_and(r < self.range_max, r > self.protect_range)
        mask_angle = jnp.logical_and(angle > 0, angle < self.azimuth_fov)
        mask = jnp.logical_and(mask_range, mask_angle)

        image = sig_at_r * jnp.exp(-1j * 2 * k * r)
        image = image.reshape(n, m)

        amp, phase = jnp.abs(image), jnp.angle(image)
        amp_db = 10 * jnp.log10(amp)
        prob = 1 - jnp.exp(-(amp_db**2) / (2 * self.sigma_amp**2))
        phase = phase.reshape(n, m, 1)
        phase = jnp.tile(phase, (1, 1, l))
        prob_phase = vonmises.pdf(self.phase_level - phase, self.k_phase)
        prob_phase = prob_phase / jnp.max(prob_phase, axis=-1, keepdims=True)

        prob_level = prob_phase * prob[:, :, None] / 2 + 0.5

        zeta = jnp.zeros((n, m, 1))

        loss_fn = partial(self.adjust_loss, prob=prob_level)
        for i in range(8):
            grad = jax.numpy.nan_to_num(jax.grad(loss_fn)(zeta))
            zeta = zeta - 5e-2 * grad

        p = jnp.clip(prob_level + zeta + self.xi, 0, 1)
        log_odds = jnp.log(p / (1 - p + 1e-7))

        return log_odds * mask.reshape(n, m, 1)

    def update_patch(
        self,
        pose_tx: Float[Array, "4 4"],
        pose_rx: Float[Array, "4 4"],
        sig: Complex[Array, "n_samples"],
    ) -> tuple[Float[Array, "2"], Float[Array, "2s 2s l"]]:
        pos_xy = (pose_tx[:2, 3] + pose_rx[:2, 3]) / 2
        idx = ((pos_xy - self.map.min_xy) // self.proc_range_res).astype(int)
        img_space = jax.lax.dynamic_slice(
            self.map.grid,
            (idx[0] - self.map.patch_radius, idx[1] - self.map.patch_radius, 0),
            (
                2 * self.map.patch_radius,
                2 * self.map.patch_radius,
                self.map.grid.shape[-1],
            ),
        )
        log_odds = self.project_sig2D(pose_tx, pose_rx, sig, img_space, self.k)
        return idx, log_odds

    def map_update(
        self,
        prob_level: Float[Array, "w h l"],
        u: tuple[Float[Array, "2"], Float[Array, "2s 2s l"]],
    ) -> tuple[dict, None]:
        idx, log_odds = u

        # Get the slice of current state
        slice_idx = (idx[0] - self.map.patch_radius, idx[1] - self.map.patch_radius, 0)

        current_logodds = jax.lax.dynamic_slice(
            prob_level,
            slice_idx,
            (2 * self.map.patch_radius, 2 * self.map.patch_radius, log_odds.shape[-1]),
        )

        # Update complex sum
        new_logodds = current_logodds + log_odds

        # Update the full maps
        prob_level = jax.lax.dynamic_update_slice(prob_level, new_logodds, slice_idx)

        return prob_level, None

    @cached_property
    def update_batch(self):
        @jax.jit
        def __inner__(
            poses_tx: Float[Array, "b 4 4"],
            poses_rx: Float[Array, "b 4 4"],
            sigs: Complex[Array, "b n_samples"],
            prob_level: Float[Array, "w h l"],
        ) -> PyTree:

            idxs, log_odds = jax.vmap(self.update_patch)(poses_tx, poses_rx, sigs)

            prob_level, _ = jax.lax.scan(self.map_update, prob_level, (idxs, log_odds))
            return prob_level

        return __inner__
