"""Occupancy SAR mapping module."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float
from mmwcas.dataset import ChirpPoseDataset
from mmwcas.process import RangeAzimuthProc

from .map import RadarMap


class OccupancySAR:
    """Occupancy SAR mapping class.

    Args:
        map: RadarMap object.
        n_sig_thresh: Number of significant observations threshold.
        amp_sigma: Amplitude sigma for Rayleigh distribution.
        range_res: Range resolution in meters.
        range_min: Minimum range in meters.
        range_max: Maximum range in meters.
        ang_res: Angular resolution in degrees.
        ang_fov: Angular field of view in degrees.
        prob_hit: Probability of hit for occupancy update.
        prob_miss: Probability of miss for occupancy update.
        clamp_log_max: Maximum log-odds value for clamping.
        clamp_log_min: Minimum log-odds value for clamping.
    """

    def __init__(
        self,
        map: RadarMap,
        n_sig_thresh: int,
        amp_sigma: float,
        range_res: float,
        range_min: float,
        range_max: float,
        ang_res: float = 0.5,
        ang_fov: float = 20.0,
        prob_hit: float = 0.7,
        prob_miss: float = 0.4,
        clamp_log_max: float = 3.5,  # 0.97
        clamp_log_min: float = -2.0,  # 0.12
    ) -> None:
        self.map = map
        self.a_res = ang_res
        self.r_res = range_res
        self.r_min = range_min
        self.r_max = range_max
        self.n_sig_thresh = n_sig_thresh
        self.amp_sigma = amp_sigma

        rs = np.arange(range_min, range_max, range_res)
        thetas = np.arange(-ang_fov, ang_fov, ang_res)
        ra = np.stack(np.meshgrid(rs, thetas, indexing="ij"), axis=-1)
        xs = ra[..., 0] * np.cos(np.deg2rad(ra[..., 1]))
        ys = ra[..., 0] * np.sin(np.deg2rad(ra[..., 1]))
        zs, ones = np.zeros_like(xs), np.ones_like(xs)
        self.local_grid = np.stack((xs, ys, zs, ones), axis=-1)

        self.origin = map.min_xy
        self.sin_res = np.sin(np.deg2rad(ang_res))
        self.ang_size = len(thetas)

        self.log_map = jnp.zeros(map.grid.shape[:2], dtype=jnp.float32)
        self.prob_hit = prob_hit
        self.prob_miss = prob_miss
        self.clamp_log_max = clamp_log_max
        self.clamp_log_min = clamp_log_min

    def __call__(
        self,
        pose: Float[Array, "4 4"],
        map_state: dict,
        cur_map: Float[Array, "H W"],
    ) -> Float[Array, "H W"]:
        """Update the occupancy map based on the given pose and map state.

        Args:
            pose: Current sensor pose.
            map_state: Current radar map state.
            cur_map: Current occupancy map.

        Returns:
            Updated occupancy map.
        """
        # calculate probability and local RA map
        map_abs = jnp.abs(map_state["complex"])
        mask = map_state["n_obs"] > self.n_sig_thresh
        map_norm = mask * (map_abs / (map_state["n_obs"] + 1))

        g_pos = jnp.matmul(pose, self.local_grid.reshape(-1, 4).T).T
        g_pos = g_pos[:, :2]
        indx_pos = (g_pos - self.origin) / self.map.resolution

        ra_val = jax.scipy.ndimage.map_coordinates(map_norm, indx_pos.T, order=1)
        ra_val = ra_val.reshape(self.local_grid.shape[:2])
        prob = 1 - jnp.exp(-(ra_val**2) / (2 * self.amp_sigma**2))

        alpha = prob
        w, h = alpha.shape
        transmitance = jnp.cumprod(
            jnp.concatenate([jnp.ones((1, h)), 1.0 - alpha + 1e-7], axis=0), axis=0
        )[:-1]
        transmitance = jnp.where(transmitance < 0.2, 0, transmitance)

        # apply density rule to the probability map
        prob = jnp.clip(prob, self.prob_miss, self.prob_hit)
        log_odds = jnp.log(prob / (1 - prob + 1e-7))
        log_odds = log_odds * transmitance

        idx = ((pose[:2, 3] - self.map.min_xy) // self.r_res).astype(int)
        img_space = jax.lax.dynamic_slice(
            self.map.grid,
            (idx[0] - self.map.patch_radius, idx[1] - self.map.patch_radius, 0),
            (
                2 * self.map.patch_radius,
                2 * self.map.patch_radius,
                self.map.grid.shape[-1],
            ),
        )
        global_c = img_space.reshape(-1, 3)
        global_c = jnp.hstack((global_c, jnp.ones((global_c.shape[0], 1))))
        R, t = pose[:3, :3], pose[:3, 3:]
        inv_p = jnp.vstack((jnp.hstack((R.T, -R.T @ t)), jnp.array([[0, 0, 0, 1]])))
        local_c = (inv_p @ global_c.T).T[:, :2]
        local_r = jnp.linalg.norm(local_c, axis=-1)
        local_t = jnp.arctan2(local_c[..., 1], local_c[..., 0])
        indx_a = (jnp.rad2deg(local_t) / self.a_res) + (self.ang_size / 2)
        indx_r = (local_r - self.r_min) / self.r_res
        val = jax.scipy.ndimage.map_coordinates(log_odds, (indx_r, indx_a), order=1)
        val = val.reshape(img_space.shape[:2])

        # protect range mask
        protect_mask = (local_r <= self.r_min).reshape(img_space.shape[:2])
        val = jnp.where(protect_mask, -2, val)

        slice_idx = (idx[0] - self.map.patch_radius, idx[1] - self.map.patch_radius)
        mval = jax.lax.dynamic_slice(
            cur_map,
            slice_idx,
            (2 * self.map.patch_radius, 2 * self.map.patch_radius),
        )

        cur_map = jax.lax.dynamic_update_slice(cur_map, mval + val, slice_idx)
        cur_map = jnp.clip(cur_map, self.clamp_log_min, self.clamp_log_max)
        return cur_map
