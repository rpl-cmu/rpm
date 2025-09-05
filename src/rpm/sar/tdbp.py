"""Back Projection SAR Imaging Module."""

from functools import cached_property
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Int, PyTree

from ..mmwcas.dataset import ChirpPoseDataset
from ..mmwcas.process import PatternAZI
from .map import RadarMap


class BackProjection:
    """Back Projection SAR Imaging Module.

    Args:
        dataset: ChirpPoseDataset.
        map_extent: Map extent in meters.
        protect_range: Minimum range to consider in meters.
        azimuth_fov: Azimuth field of view in degrees.
        smooth_window: Window size for amplitude smoothing.
        resolution_scale: Scale factor for range resolution.
        resolution: Map resolution in meters.
        amp_sigma: Standard deviation for amplitude Gaussian smoothing.
    """

    def __init__(
        self,
        dataset: ChirpPoseDataset,
        map_extent: float = 10,  # meter
        protect_range: float = 0.5,  # meter
        azimuth_fov: float = 20,  # degree
        smooth_window: int = 1,
        resolution_scale: int = 1,
        resolution: float = 0.1,
        amp_sigma: float = 0.5,
    ) -> None:
        # frame poses
        poses = dataset.chirp_poses[:, 0, 0, :, :]

        # signal parameters
        param = dataset.adc.param
        self.range_res = param.rangeResolution
        self.range_max = min(map_extent, param.maxRange)
        C = param.speedOfLight
        F0 = param.startFreqConst
        self.k = 2 * jnp.pi * (F0 / C)
        self.window = jnp.hanning(param.numADCSample)

        self.amp_smooth = True if smooth_window > 1 else False
        self.smooth_kernel = jnp.ones(smooth_window) / smooth_window

        # radiation pattern parameters
        self.pattern = PatternAZI(xla=jnp)
        self.azimuth_fov = azimuth_fov

        # map parameters
        self.protect_range = protect_range
        self.proc_range_res = resolution
        self.sig_range_res = param.rangeResolution / resolution_scale
        self.fft_len = param.numADCSample * resolution_scale
        self.map = RadarMap(
            resolution=self.proc_range_res,
            poses=poses,
            map_extent=map_extent,
            amp_sigma=amp_sigma,
        )

    def project_sig2d(
        self,
        pose_tx: Float[Array, "4 4"],
        pose_rx: Float[Array, "4 4"],
        sig: Complex[Array, "n_samples"],
        grid: Float[Array, "n m 3"],
        k: float,
    ) -> Complex[Array, "n m"]:
        """Project 1D signal to 2D image grid using back projection.

        Args:
            pose_tx: Transmitter pose matrix.
            pose_rx: Receiver pose matrix.
            sig: 1D complex signal array.
            grid: image grid points.
            k: Wave number.

        Returns:
            complex image array.
        """
        pts = pose_tx[:3, 3]
        pixels = grid.reshape(-1, 3)
        ray_v = jnp.dot(pose_tx[:3, :3], jnp.array([1, 0, 0]))

        rtx = jnp.linalg.norm(pixels - pose_tx[:3, 3], axis=-1)
        rrx = jnp.linalg.norm(pixels - pose_rx[:3, 3], axis=-1)
        r = (rtx + rrx) * 0.5
        sig = sig - jnp.mean(sig)
        sig_fft = jnp.fft.fft(self.window * sig, n=self.fft_len, norm="forward")

        if self.amp_smooth:
            mag = jnp.abs(sig_fft)
            mag_smooth = jnp.convolve(mag, self.smooth_kernel, mode="same")
            sig_fft = sig_fft / mag * mag_smooth

        idx = (r // self.sig_range_res).astype(int)
        sig_min, sig_max = sig_fft[idx], sig_fft[idx + 1]
        sig_at_r = sig_min + (r - idx * self.sig_range_res) / self.sig_range_res * (
            sig_max - sig_min
        )

        r_plane = jnp.linalg.norm(pixels[:, :2] - pts[:2], axis=-1)
        ray_dot = (pixels[:, :2] - pts[:2]) @ ray_v[:2] / r_plane
        angle = jnp.rad2deg(jnp.arccos(ray_dot))
        gain = jnp.power(10, self.pattern.gain(angle) / 10)

        mask_range = jnp.logical_and(r < self.range_max, r > self.protect_range)
        mask_angle = jnp.logical_and(angle > 0, angle < self.azimuth_fov)
        mask = jnp.logical_and(mask_range, mask_angle)

        image = sig_at_r * jnp.exp(-1j * 2 * k * r) * gain * mask
        image = image.reshape(grid.shape[:2])

        return image

    def update_patch(
        self,
        pose_tx: Float[Array, "4 4"],
        pose_rx: Float[Array, "4 4"],
        sig: Complex[Array, "n_samples"],
    ) -> tuple[Int[Array, "2"], Complex[Array, "h w"]]:
        """Update image patch for a single chirp.

        Args:
            pose_tx: Transmitter pose matrix.
            pose_rx: Receiver pose matrix.
            sig: 1D complex signal array.

        Returns:
            Index of the image patch.
            Image patch update.
        """
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
        img_update = self.project_sig2d(pose_tx, pose_rx, sig, img_space, self.k)
        return idx, img_update

    def map_update(
        self,
        state: dict,
        u: tuple[Int[Array, "2"], Complex[Array, "h w"]],
    ) -> tuple[dict, None]:
        """Update the radar map state with a new image patch.

        Args:
            state: Current radar map state.
            u: Tuple of index and image patch.

        Returns:
            Updated radar map state.
        """
        idx, img = u

        # Get the slice of current state
        slice_idx = (idx[0] - self.map.patch_radius, idx[1] - self.map.patch_radius)

        slice_state = lambda key: jax.lax.dynamic_slice(
            state[key],
            slice_idx,
            (2 * self.map.patch_radius, 2 * self.map.patch_radius),
        )
        current_complex = slice_state("complex")
        current_sin_sum = slice_state("sin_sum")
        current_cos_sum = slice_state("cos_sum")
        current_n_obs = slice_state("n_obs")
        current_update = slice_state("update")

        # Update complex sum
        new_complex = current_complex + img

        # Update phase statistics where we have valid measurements
        valid_mask = jnp.abs(img) > 0
        phase = jnp.angle(img)

        # Accumulate sin and cos components for phase variance
        new_sin_sum = current_sin_sum + jnp.where(valid_mask, jnp.sin(phase), 0)
        new_cos_sum = current_cos_sum + jnp.where(valid_mask, jnp.cos(phase), 0)
        new_n_obs = current_n_obs + valid_mask.astype(jnp.int32)

        new_update = jnp.logical_or(current_update, valid_mask)

        # Update the full maps
        update_slice = lambda key, new_val: jax.lax.dynamic_update_slice(
            state[key], new_val, slice_idx
        )
        state["complex"] = update_slice("complex", new_complex)
        state["sin_sum"] = update_slice("sin_sum", new_sin_sum)
        state["cos_sum"] = update_slice("cos_sum", new_cos_sum)
        state["n_obs"] = update_slice("n_obs", new_n_obs)
        state["update"] = update_slice("update", new_update)

        return state, None

    @cached_property
    def update_batch(self) -> Callable:
        @jax.jit
        def inner(
            poses_tx: Float[Array, "b 4 4"],
            poses_rx: Float[Array, "b 4 4"],
            sigs: Complex[Array, "b n_samples"],
            state: dict,
        ) -> PyTree:
            """Update radar map state with a batch of chirps.

            Args:
                poses_tx: Transmitter pose matrices.
                poses_rx: Receiver pose matrices.
                sigs: 1D complex signal arrays.
                state: Current radar map state.

            Returns:
                Updated radar map state.
            """
            state["update"] = jnp.zeros_like(state["update"], dtype=jnp.bool_)
            idxs, imgs = jax.vmap(self.update_patch)(poses_tx, poses_rx, sigs)
            state, _ = jax.lax.scan(self.map_update, state, (idxs, imgs))
            return state

        return inner
