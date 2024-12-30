import numpy as np
from functools import cached_property

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Complex, Float, Array, Int, Bool
from sar import ChirpPoseDataset, RadarMap
from mmwcas.mmwcas import PatternAZI


class AFBackProjection:
    def __init__(
        self,
        dataset: ChirpPoseDataset,
        map_extent: float = 10,  # meter
        protect_range: float = 0.5,  # meter
        azimuth_fov: float = 40,  # degree
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

        # radiation pattern parameters
        self.pattern = PatternAZI("mmwcas/pattern.mat")
        self.azimuth_fov = azimuth_fov

        # map parameters
        self.protect_range = protect_range
        self.proc_range_res = param.rangeResolution / 2
        # self.proc_range_res = self.range_res
        self.map = RadarMap(
            resolution=self.proc_range_res,
            pos_max_xy=np.max(poses[:, :2, 3], axis=0),
            pos_min_xy=np.min(poses[:, :2, 3], axis=0),
            map_extent=map_extent,
            protect_range=protect_range,
        )

    def project_sig2D(
        self,
        pose_tx: Float[Array, "4 4"],
        t_rx: Float[Array, "3"],
        sig: Complex[Array, "n_samples"],
        grid: Float[Array, "n m 3"],
        k: Complex[Array, " "],
    ) -> Complex[Array, "n m"]:
        pts = pose_tx[:3, 3]
        pixels = grid.reshape(-1, 3)

        ray_v = jnp.dot(pose_tx[:3, :3], jnp.array([1, 0, 0]))
        ray = pixels - pts
        rtx = jnp.linalg.norm(pixels - pose_tx[:3, 3], axis=-1)
        rrx = jnp.linalg.norm(pixels - t_rx, axis=-1)
        r = (rtx + rrx) * 0.5
        sig_fft = jnp.fft.fft(jnp.hanning(sig.size) * sig)
        idx = (r // self.range_res).astype(int)
        sig_min, sig_max = sig_fft[idx], sig_fft[idx + 1]
        sig_at_r = sig_min + (r % self.range_res) * (sig_max - sig_min) / self.range_res
        image = sig_at_r * jnp.exp(-1j * 2 * k * r)

        image = image.reshape(grid.shape[:2])
        ray_dot = ray @ ray_v / r
        angle = jnp.rad2deg(jnp.arccos(ray_dot))
        # gain = self.pattern.gain(angle)
        # image = image * gain.reshape(grid.shape[:2])

        mask_range = jnp.logical_and(r < self.range_max, r > self.protect_range)
        mask_angle = jnp.logical_and(angle > 0, angle < self.azimuth_fov)
        mask = jnp.logical_and(mask_range, mask_angle).reshape(grid.shape[:2])

        image = image * mask

        return image

    def update_patch(
        self,
        pose_tx: Float[Array, "4 4"],
        t_rx: Float[Array, "3"],
        sig: Complex[Array, "n_samples"],
    ) -> tuple[Float[Array, "2"], Complex[Array, "2s 2s"]]:
        pos_xy = (pose_tx[:2, 3] + t_rx[:2]) / 2
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
        img_update = self.project_sig2D(pose_tx, t_rx, sig, img_space, self.k)
        return idx, img_update

    def map_update(
        self,
        state: dict,
        u: tuple[Float[Array, "2"], Complex[Array, "2s 2s"]],
    ) -> tuple[dict, None]:
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
    def update_batch(self):
        @jax.jit
        def __inner__(
            poses_tx: Float[Array, "b 4 4"],
            poses_rx: Float[Array, "b 4 4"],
            sigs: Complex[Array, "b n_samples"],
            state: dict,
        ) -> PyTree:
            state["update"] = jnp.zeros_like(state["update"], dtype=jnp.bool_)
            idxs, imgs = jax.vmap(self.update_patch)(poses_tx, poses_rx, sigs)
            state, _ = jax.lax.scan(self.map_update, state, (idxs, imgs))
            return state

        return __inner__

    def batch_project(
        self,
        poses_tx: Float[Array, "b 4 4"],
        t_rx: Float[Array, "b 3"],
        sigs: Complex[Array, "b n_samples"],
        state: dict,
    ) -> PyTree:
        state["update"] = jnp.zeros_like(state["update"], dtype=jnp.bool_)
        idxs, imgs = jax.vmap(self.update_patch)(poses_tx, t_rx, sigs)
        state, _ = jax.lax.scan(self.map_update, state, (idxs, imgs))
        return state
