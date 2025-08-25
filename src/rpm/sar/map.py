"""Radar Map module for Synthetic Aperture Radar (SAR) processing."""

from __future__ import annotations

import pickle

import imageio
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float


class RadarMap:
    """Radar Map class for SAR processing.

    Args:
        resolution: Map resolution in meters.
        poses: Array of sensor poses.
        map_extent: Extent of the map around the sensor positions in meters.
        amp_sigma: Amplitude sigma for Rayleigh distribution.
        block: Block size for padding the map dimensions.
    """

    def __init__(
        self,
        resolution: float,
        poses: np.ndarray,
        map_extent: float = 10,  # meter
        amp_sigma=0.5,
        block: int = 16,
    ):
        # map parameters
        self.resolution = resolution
        self.map_extent = map_extent
        self.amp_sigma = amp_sigma
        self.patch_radius = int(map_extent // resolution) - 1

        pos_max_xy = np.max(poses[:, :2, 3], axis=0)
        pos_min_xy = np.min(poses[:, :2, 3], axis=0)
        self.max_xy = pos_max_xy + map_extent
        self.min_xy = pos_min_xy - map_extent
        w = int((self.max_xy[0] - self.min_xy[0]) // resolution)
        h = int((self.max_xy[1] - self.min_xy[1]) // resolution)
        pad_w, pad_h = block - w % block, block - h % block
        self.max_xy[0] += pad_w * resolution
        self.max_xy[1] += pad_h * resolution
        xs = np.linspace(
            self.min_xy[0],
            self.max_xy[0],
            int((self.max_xy[0] - self.min_xy[0]) // resolution),
        )
        ys = np.linspace(
            self.min_xy[1],
            self.max_xy[1],
            int((self.max_xy[1] - self.min_xy[1]) // resolution),
        )

        pos = poses[:, :3, 3]
        pos_center = np.mean(pos, axis=0)
        pos = pos - pos_center
        _, _, vh = np.linalg.svd(pos, full_matrices=False)
        normal = vh[-1, :]
        self.grid = np.stack(np.meshgrid(xs, ys, [0], indexing="ij"), axis=-1)
        self.grid = self.grid.squeeze()

        # internal state
        self.complex = np.zeros(self.grid.shape[:2], dtype=np.complex64)
        self.sin_sum = np.zeros(self.grid.shape[:2], dtype=np.float32)
        self.cos_sum = np.zeros(self.grid.shape[:2], dtype=np.float32)
        self.n_obs = np.zeros(self.grid.shape[:2], dtype=np.int32)
        w, h, _ = self.grid.shape

        print("map size: ", w, h)

    def get_map(
        self,
    ) -> dict[str, np.ndarray]:
        """Return map internal states.

        Returns:
            A dictionary containing the map's internal states:
                - complex: Complex map values.
                - sin_sum: Sum of sine of phases.
                - cos_sum: Sum of cosine of phases.
                - n_obs: Number of observations per pixel.
                - update: Boolean array indicating updated pixels.
        """
        update = np.zeros_like(self.n_obs, dtype=np.bool_)
        return {
            "complex": self.complex,
            "sin_sum": self.sin_sum,
            "cos_sum": self.cos_sum,
            "n_obs": self.n_obs,
            "update": update,
        }

    def update_map(self, map_state: dict) -> None:
        """Update map internal states from a given state dictionary.

        Args:
            map_state: A dictionary containing the map's internal states to update.
        """
        for key, val in map_state.items():
            self.__dict__[key] = val

    def save(self, save_dir: str) -> None:
        """Save the RadarMap object to a specified directory."""
        with open(f"{save_dir}/map.pkl", "wb") as f:
            pickle.dump(self, f)

    def save_probmap(
        self, save_path: str, prob_map: np.ndarray
    ) -> tuple[Float[np.ndarray, "h w"], Float[np.ndarray, "3"], float]:
        """Save the probability map to a specified path.

        Args:
            save_path: Path to save the probability map.
            prob_map: Probability map to be saved.

        Returns:
                The saved probability map data
                translation vector,
                map resolution.
        """
        data = prob_map
        resolution = self.resolution
        h, w = self.grid.shape[:2]
        translation = np.asarray([self.grid[0, 0, 0], self.grid[0, 0, 1], 0])
        rotation = np.asarray([0.0, 0.0, 0.0, 0.0])
        map_data = {
            "data": data,
            "resolution": resolution,
            "width": w,
            "height": h,
            "t": translation,
            "r": rotation,
        }
        with open(save_path, "wb") as f:
            pickle.dump(map_data, f)

        return data, translation, resolution

    @staticmethod
    def load(load_dir: str) -> RadarMap:
        """Load a RadarMap object from a specified directory."""
        with open(f"{load_dir}/map.pkl", "rb") as f:
            return pickle.load(f)

    def get_phase_variance(
        self, map_state: dict, eps: float = 1e-10
    ) -> Float[np.ndarray, "h w"]:
        """Compute the variance of circular standard deviation from accumulated phase statistics.

        Args:
            map_state: A dictionary containing the map's internal states.
            eps: A small value to avoid division by zero.

        Returns:
            Variance of circular standard deviation for each pixel.
        """
        valid_obs = map_state["n_obs"] > 1

        # Compute mean direction and length of the mean resultant vector (R)
        mean_sin = np.where(
            valid_obs, map_state["sin_sum"] / (map_state["n_obs"] + eps), 0
        )
        mean_cos = np.where(
            valid_obs, map_state["cos_sum"] / (map_state["n_obs"] + eps), 0
        )
        R = np.sqrt(mean_sin**2 + mean_cos**2)

        # Variance of circular standard deviation = -2 * ln(R)
        variance = np.where(valid_obs, -2 * np.log(np.clip(R, eps, 1.0)), 0)

        return variance

    def visualize(self, save_dir: str, color_map: str = "hot") -> None:
        """Visualize and save different representations of the radar map.

        Args:
            save_dir: Directory to save the visualizations.
            color_map: Colormap to use for visualizations.
        """
        map_state = self.get_map()
        cmap = plt.get_cmap(color_map)

        def to_png(save_dir: str, map: np.ndarray, name: str):
            map = cmap(map)[:, :, :3] * 255
            imageio.imwrite(f"{save_dir}/{name}", map[:, :, ::-1])

        map_abs = np.abs(map_state["complex"])
        valid = map_state["n_obs"] > 0

        # decibel
        map_dB = np.copy(map_abs)
        map_dB[valid] = 10 * np.log10(map_dB[valid])
        map_dB[valid] = map_dB[valid] / np.max(map_dB[valid])
        to_png(save_dir, map_dB, "map_dB.png")

        # clip
        left, right = np.percentile(map_abs, np.array([0.0, 99.0]))
        map_clip = (np.clip(map_abs, left, right) - left) / (right - left)
        map_clip = map_clip / np.max(map_clip)
        to_png(save_dir, map_clip, "map_clip.png")

        # rayleigh
        map_norm = map_abs / (map_state["n_obs"] + 1)
        map_norm *= map_state["n_obs"] > 64
        prob = 1 - np.exp(-(map_norm**2) / (2 * self.amp_sigma**2))
        to_png(save_dir, prob, "map_rayleigh.png")

        # phase
        map_phase = np.angle(map_state["complex"])
        map_phase = (map_phase + np.pi) / (2 * np.pi)
        hsv_cmap = plt.get_cmap("hsv")
        map_phase = hsv_cmap(map_phase)[:, :, :3] * 255
        imageio.imwrite(f"{save_dir}/map_phase.png", map_phase[:, :, ::-1])

        # variance
        map_variance = self.get_phase_variance(map_state)
        map_variance = map_variance / np.max(map_variance)
        to_png(save_dir, map_variance, "map_phase_var.png")

        # Histogram equalization
        abs_sorted = np.sort(map_abs[map_state["n_obs"] > 0].reshape(-1))
        cdf = np.cumsum(abs_sorted)
        cdf = cdf / cdf[-1]

        # equalized
        new_map_idx = np.searchsorted(
            abs_sorted, map_abs[map_state["n_obs"] > 0].reshape(-1)
        )
        new_map = cdf[new_map_idx]
        map_abs[map_state["n_obs"] > 0] = new_map
        to_png(save_dir, map_abs, "map_equalized.png")
