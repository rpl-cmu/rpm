import time
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin

import cv2
import open3d as o3d


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