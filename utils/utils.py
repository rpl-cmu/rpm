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
