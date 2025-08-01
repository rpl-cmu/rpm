import cv2
import numpy as np
from scipy.optimize import least_squares



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
