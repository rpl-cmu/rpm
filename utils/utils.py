import cv2
import numpy as np
from scipy.optimize import least_squares


def bodyframe_vel_estimate(
    radar_points: np.ndarray,
    vs_init: np.ndarray = np.array([1, 0, 0]),
    threshold=0.5,
):
    doppler_v = radar_points[:, -1]
    norm_r = np.linalg.norm(radar_points[:, :3], axis=1)
    unit_r = -radar_points[:, :3] / norm_r[:, None]

    # cauchy loss doppler residual
    lstsq_func = lambda x, A, b: A @ x - b
    res_log = least_squares(
        lstsq_func, vs_init, loss="cauchy", f_scale=1, args=(unit_r, doppler_v)
    )
    vs = res_log.x

    res = unit_r @ vs - doppler_v
    mask = np.abs(res) < threshold

    return vs, radar_points[mask]


def map_pc(
    grid: np.ndarray,
    resolution: float,
    pc: np.ndarray,
    save_dir: str,
    name: str = "map_pc.png",
) -> None:
    w, h = grid.shape[:2]
    idx = np.round((pc[:, :2] - grid[0, 0, :2]) / resolution).astype(int)
    map2D = np.ones((w, h, 3)) * 255
    idx = idx[np.logical_and(idx[:, 0] >= 0, idx[:, 1] >= 0)]
    idx = idx[np.logical_and(idx[:, 0] < w, idx[:, 1] < h)]
    map2D[idx[:, 0], idx[:, 1], :] = 0
    cv2.imwrite(f"{save_dir}/{name}", map2D.astype(np.uint8))
    return grid[map2D[..., 0] == 0], map2D


def map_to_pts(
    prob_map: np.ndarray, origin: np.ndarray, res: float, thresh_prob: float = 0.6
):
    xs = np.arange(prob_map.shape[0]) * res + origin[0]
    ys = np.arange(prob_map.shape[1]) * res + origin[1]
    pts = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
    pts = pts.reshape(-1, 2)

    prob_map = prob_map.reshape(-1)
    pts = pts[prob_map > thresh_prob]
    pts = np.hstack((pts, np.zeros((pts.shape[0], 1))))
    return pts
