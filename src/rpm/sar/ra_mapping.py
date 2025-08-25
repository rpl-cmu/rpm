"""Range-Azimuth mapping."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float
from mmwcas.dataset import ChirpPoseDataset
from mmwcas.process import RangeAzimuthProc

from .map import RadarMap


class RAmapping:
    """Range-Azimuth mapping for SAR processing.

    Args:
        dataset: ChirpPoseDataset containing chirp signals and poses.
        map_extent: Extent of the map around the sensor positions in meters.
        resolution: Map resolution in meters.
        protect_range: Minimum range to protect from updates in meters.
        angle_3db: 3db beamwidth angle in degrees.
        amp_sigma: Amplitude sigma for Rayleigh distribution.
        prob_hit: Probability of hit for occupancy mapping.
        prob_miss: Probability of miss for occupancy mapping.
        clamp_log_max: Maximum log-odds value for clamping.
        clamp_log_min: Minimum log-odds value for clamping.
        angele_fft_size: Size of the angle FFT.
    """

    def __init__(
        self,
        dataset: ChirpPoseDataset,
        map_extent: float = 10,  # meter
        resolution: float = 0.1,
        protect_range: float = 0.3,
        angle_3db: float = 20.0,
        amp_sigma: float = 2.0,
        prob_hit: float = 1,
        prob_miss: float = 0.45,
        clamp_log_max: float = 3.5,  # 0.97
        clamp_log_min: float = -2.0,  # 0.12
        angele_fft_size: int = 256,
    ) -> None:
        # frame poses
        poses = dataset.chirp_poses[:, 0, 0, :, :]
        self.param = dataset.adc.param

        # map parameters
        self.proc_range_res = resolution
        self.map = RadarMap(
            resolution=self.proc_range_res,
            poses=poses,
            map_extent=map_extent,
        )
        self.map_val = jnp.zeros(self.map.grid.shape[:2], dtype=jnp.float32)
        self.range_bin_min = int(protect_range / self.param.rangeBinSize) - 1
        range_bin_max = int(map_extent / self.param.rangeBinSize) + 1
        self.proc = RangeAzimuthProc(
            self.param,
            angele_fft_size=angele_fft_size,
            range_bin_min=self.range_bin_min,
            range_bin_max=range_bin_max,
            angle_3db=angle_3db,
            output_normalize=False,
        )
        self.range_mask = np.ones(range_bin_max, dtype=bool)
        self.range_mask[: self.range_bin_min] = False

        cartisian = jnp.stack((self.proc.x_axis, self.proc.y_axis), axis=-1)
        r = jnp.linalg.norm(cartisian, axis=-1)
        theta = jnp.arctan2(cartisian[..., 1], cartisian[..., 0])
        self.polar = jnp.stack((r, theta), axis=-1)
        self.min_t, self.max_t = jnp.min(theta), jnp.max(theta)
        self.min_r, self.max_r = protect_range, map_extent

        afft_size = self.proc.azimuth_proc.angle_fft_size
        self.sin_res = 2.0 / (afft_size - 1)
        self.ang_size = cartisian.shape[1]
        self.range_res = self.param.rangeBinSize

        self.amp_sigma = amp_sigma
        self.prob_hit = prob_hit
        self.prob_miss = prob_miss
        self.clamp_log_max = clamp_log_max
        self.clamp_log_min = clamp_log_min

    def __call__(
        self,
        pose: Float[Array, "4 4"],
        sig_tensor: Complex[Array, "sample chirp rx tx"],
        cur_map: Float[Array, "H W"],
    ) -> Float[Array, "H W"]:
        """Update the RA map with a new chirp signal and pose.

        Args:
            pose: 4x4 sensor pose matrix.
            sig_tensor: Complex chirp signal tensor.
            cur_map: Current log-odds map to be updated.

        Returns:
            Updated log-odds map.
        """
        # calculate local RA map and probability
        ra = self.proc(sig_tensor)
        prob = 1 - jnp.exp(-(ra**2) / (2 * self.amp_sigma**2))
        ang_mask = jnp.max(ra, axis=0) < 0.5
        prob = prob * (1 - ang_mask[None, :]) + 0.5 * ang_mask[None, :]

        # model density from RA values
        density = jnp.clip(jnp.log10(ra), 0, jnp.inf)
        alpha = 1 - jnp.exp(-density)
        w, h = alpha.shape
        transmittance = jnp.cumprod(
            jnp.concatenate([jnp.ones((1, h)), 1.0 - alpha + 1e-7], axis=0), axis=0
        )[:-1]

        # apply density to the log odds
        prob = jnp.clip(prob, self.prob_miss, self.prob_hit)
        log_odds = jnp.log(prob / (1 - prob + 1e-7))
        log_odds = transmittance * log_odds

        # map value to map cells
        idx = ((pose[:2, 3] - self.map.min_xy) // self.proc_range_res).astype(int)
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

        mask_r = local_r < self.max_r
        mask_t = jnp.logical_and(local_t > self.min_t, local_t < self.max_t)
        mask = jnp.logical_and(mask_r, mask_t)

        local_s = jnp.sin(local_t)
        indx_a = (local_s / self.sin_res) + (self.ang_size // 2)
        indx_r = (local_r / self.range_res) - self.range_bin_min

        val = jax.scipy.ndimage.map_coordinates(log_odds, (indx_r, indx_a), order=1)
        val = (val * mask).reshape(img_space.shape[:2])

        # # protect range mask
        protect_mask = (local_r <= self.min_r).reshape(img_space.shape[:2])
        val = jnp.where(protect_mask, -2, val)

        # update log_odds map
        slice_idx = (idx[0] - self.map.patch_radius, idx[1] - self.map.patch_radius)
        mval = jax.lax.dynamic_slice(
            cur_map,
            slice_idx,
            (2 * self.map.patch_radius, 2 * self.map.patch_radius),
        )

        cur_map = jax.lax.dynamic_update_slice(cur_map, mval + val, slice_idx)
        cur_map = jnp.clip(cur_map, self.clamp_log_min, self.clamp_log_max)

        return cur_map
