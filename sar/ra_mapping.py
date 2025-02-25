import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float
from mmwcas.dataset import ChirpPoseDataset
from mmwcas.process import RangeAzimuthProc
from .map import RadarMap


class RAmapping:
    def __init__(
        self,
        dataset: ChirpPoseDataset,
        map_extent: float = 10,  # meter
        resolution_scale: int = 2,
        angele_fft_size=256,
        range_bin_min=5,
        range_bin_max=236,
        angle_3dB=30.0,
        normalize_factor=10,
    ) -> None:
        # frame poses
        poses = dataset.chirp_poses[:, 0, 0, :, :]
        self.param = dataset.adc.param

        # map parameters
        self.proc_range_res = self.param.rangeResolution / resolution_scale
        self.fft_len = self.param.numADCSample * resolution_scale
        self.map = RadarMap(
            resolution=self.proc_range_res,
            poses=poses,
            map_extent=map_extent,
        )
        self.map_val = jnp.zeros(self.map.grid.shape[:2], dtype=jnp.float32)

        self.proc = RangeAzimuthProc(
            self.param,
            angele_fft_size=angele_fft_size,
            range_bin_min=range_bin_min,
            range_bin_max=range_bin_max,
            angle_3dB=angle_3dB,
            output_normalize=False,
        )

        cartisian = jnp.stack((self.proc.x_axis, self.proc.y_axis), axis=-1)
        r = jnp.linalg.norm(cartisian, axis=-1)
        theta = jnp.arctan2(cartisian[..., 1], cartisian[..., 0])
        self.polar = jnp.stack((r, theta), axis=-1)
        self.min_r, self.max_r = jnp.min(r), jnp.max(r)
        self.min_t, self.max_t = jnp.min(theta), jnp.max(theta)

        afft_size = self.proc.azimuth_proc.angle_fft_size
        self.sin_res = 2.0 / (afft_size - 1)
        self.ang_size = cartisian.shape[1]

        self.range_res = self.param.rangeBinSize
        self.range_bin_min = range_bin_min

    def __call__(
        self,
        pose: Float[Array, "4 4"],
        sig_tensor: Complex[Array, "sample chirp rx tx"],
        cur_map: Float[Array, "H W"],
    ):
        ra = self.proc(sig_tensor)
        ra_img = jnp.clip(jnp.log10(ra), -0.25, 1.0)
        density = ra / 20.0

        alpha = 1 - jnp.exp(-density)
        w, h = alpha.shape
        transmission = jnp.cumprod(
            jnp.concatenate([jnp.ones((1, h)), 1.0 - alpha + 1e-7], axis=0), axis=0
        )[:-1]

        ra_img = transmission * ra_img
        # ra_img = jnp.clip(ra / 10.0, 0, 1) - 0.05

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

        mask_r = jnp.logical_and(local_r > self.min_r, local_r < self.max_r)
        mask_t = jnp.logical_and(local_t > self.min_t, local_t < self.max_t)
        mask = jnp.logical_and(mask_r, mask_t)

        local_s = jnp.sin(local_t)
        indx_a = (local_s / self.sin_res) + (self.ang_size // 2)
        indx_r = (local_r / self.range_res) - self.range_bin_min

        val = jax.scipy.ndimage.map_coordinates(ra_img, (indx_r, indx_a), order=1)
        val = (val * mask).reshape(img_space.shape[:2])

        slice_idx = (idx[0] - self.map.patch_radius, idx[1] - self.map.patch_radius)
        mval = jax.lax.dynamic_slice(
            cur_map,
            slice_idx,
            (2 * self.map.patch_radius, 2 * self.map.patch_radius),
        )

        cur_map = jax.lax.dynamic_update_slice(cur_map, mval + val, slice_idx)
        cur_map = jnp.clip(cur_map, -2, 3.5)
        return cur_map
