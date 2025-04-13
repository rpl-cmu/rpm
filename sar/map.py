from __future__ import annotations

import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt


class RadarMap:
    def __init__(
        self,
        resolution: float,
        poses: np.ndarray,
        map_extent: float = 10,  # meter
        amp_sigma=0.5,
        block: int = 16,
        prior_cov: float = 1000000.0,
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
        x, y = self.grid[:, :, 0], self.grid[:, :, 1]
        self.grid[:, :, 2] = (
            -(normal[0] * (x - pos_center[0]) + normal[1] * (y - pos_center[1]))
            / normal[2]
            + pos_center[2]
        )

        # internal state
        self.complex = np.zeros(self.grid.shape[:2], dtype=np.complex64)
        self.sin_sum = np.zeros(self.grid.shape[:2], dtype=np.float32)
        self.cos_sum = np.zeros(self.grid.shape[:2], dtype=np.float32)
        self.n_obs = np.zeros(self.grid.shape[:2], dtype=np.int32)
        w, h, _ = self.grid.shape
        self.cov = np.zeros((w, h, 2, 2), dtype=np.float32) + prior_cov * np.eye(2)

        print("map size: ", w, h)

    def get_map(
        self,
    ) -> dict[str, np.ndarray]:
        """
        return map internal states
        """
        update = np.zeros_like(self.n_obs, dtype=np.bool_)
        return {
            "complex": self.complex,
            "sin_sum": self.sin_sum,
            "cos_sum": self.cos_sum,
            "n_obs": self.n_obs,
            "cov": self.cov,
            "update": update,
        }

    def update_map(self, map_state: dict):
        for key, val in map_state.items():
            self.__dict__[key] = val

    def save(self, save_dir: str):
        with open(f"{save_dir}/map.pkl", "wb") as f:
            pickle.dump(self, f)

    def save_probmap(self, save_path: str, prob_map: np.ndarray):
        data = prob_map
        resolution = self.resolution
        h, w = self.grid.shape[:2]
        t = [self.grid[0, 0, 0], self.grid[0, 0, 1], 0]
        r = [0.0, 0.0, 0.0, 0.0]
        map_data = {
            "data": data,
            "resolution": resolution,
            "width": w,
            "height": h,
            "t": t,
            "r": r,
        }
        with open(save_path, "wb") as f:
            pickle.dump(map_data, f)

    @staticmethod
    def load(load_dir: str) -> RadarMap:
        with open(f"{load_dir}/map.pkl", "rb") as f:
            return pickle.load(f)

    def get_phase_variance(self, map_state: dict, eps: float = 1e-10) -> np.ndarray:
        """Compute the variance of circular standard deviation from accumulated phase statistics."""
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

    def visualize(self, save_dir: str, color_map: str = "hot"):

        map_state = self.get_map()
        color_map = plt.get_cmap(color_map)

        def to_png(save_dir: str, map: np.ndarray, name: str):
            map = color_map(map)[:, :, :3] * 255
            cv2.imwrite(f"{save_dir}/{name}", map[:, :, ::-1])

        map_abs = np.abs(map_state["complex"])
        valid = map_state["n_obs"] > 0

        # decibel
        map_dB = np.copy(map_abs)
        map_dB[valid] = 10 * np.log10(map_dB[valid])
        map_dB[valid] = map_dB[valid] / np.max(map_dB[valid])
        to_png(save_dir, map_dB, "map_dB.png")

        # # normalized
        # map_norm = np.clip(map_abs / args.map_max, 0, 1)
        # to_png(save_dir, map_norm, f"map_norm_{args.map_max:.0e}.png")

        # clip
        left, right = np.percentile(map_abs, np.array([0.0, 99.0]))
        map_clip = (np.clip(map_abs, left, right) - left) / (right - left)
        map_clip = map_clip / np.max(map_clip)
        to_png(save_dir, map_clip, "map_clip.png")

        # rayleigh
        map_norm = map_abs / map_state["n_obs"]
        map_norm *= map_state["n_obs"] > 64
        prob = 1 - np.exp(-(map_norm**2) / (2 * self.amp_sigma**2))
        to_png(save_dir, prob, "map_rayleigh.png")

        # phase
        map_phase = np.angle(map_state["complex"])
        map_phase = (map_phase + np.pi) / (2 * np.pi)
        hsv_cmap = plt.get_cmap("hsv")
        map_phase = hsv_cmap(map_phase)[:, :, :3] * 255
        cv2.imwrite(f"{save_dir}/map_phase.png", map_phase[:, :, ::-1])

        # variance
        map_variance = self.get_phase_variance(map_state)
        map_variance = map_variance / np.max(map_variance)
        to_png(save_dir, map_variance, "map_variance.png")

        # # variance scaled
        # map_scaled = map_abs / map_variance
        # map_scaled = np.clip(map_scaled / args.map_max * 10, 0, 1)
        # to_png(save_dir, map_scaled, f"map_scaled_{args.map_max*10:.0e}.png")

        # pc
        thresh = np.percentile(map_abs, 95)
        mask = np.ones_like(map_abs, dtype=np.uint8) * 255
        mask[map_abs > thresh] = 0
        cv2.imwrite(f"{save_dir}/map_sar_pc.png", mask)

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

        # plt.figure()
        # plt.hist(abs_sorted, bins=1000, log=True)
        # plt.title("log histogram of map_abs")
        # plt.savefig(f"{save_dir}/map_abs_hist.png", bbox_inches="tight")

        # plt.figure()
        # plt.plot(abs_sorted, cdf)
        # plt.title("cdf of map_abs")
        # plt.savefig(f"{save_dir}/map_cdf.png", bbox_inches="tight")

        # plt.figure()
        # plt.hist(new_map, bins=1000, log=True)
        # plt.title("log histogram of equalized map_abs")
        # plt.savefig(f"{save_dir}/map_equalized_hist.png", bbox_inches="tight")

        # plt.figure()
        # new_map_sorted = np.sort(new_map)
        # cdf = np.cumsum(new_map_sorted)
        # cdf = cdf / cdf[-1]
        # plt.plot(new_map_sorted, cdf)
        # plt.title("cdf of equalized map_abs")
        # plt.savefig(f"{save_dir}/map_equalized_cdf.png", bbox_inches="tight")
