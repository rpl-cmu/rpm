"""AoA parameter class definition."""

from typing import Tuple

import numba as nb
import numpy as np
from jaxtyping import Array, Float

spec = [
    ("gamma", nb.float32),
    ("d", nb.float32),
    ("rangeBinSize", nb.float32),
    ("velocityBinSize", nb.float32),
    ("dopplerFFTSize", nb.int32),
    ("sidelobe_azi_db", nb.float32),
    ("sidelobe_ele_db", nb.float32),
    ("n_pc_preserve", nb.int32),
    ("azimuth_fov", nb.float32[:]),
    ("elevation_fov", nb.float32[:]),
    ("vec_angle", nb.float32[:]),
]


@nb.experimental.jitclass(spec)  # type: ignore
class AoAparam:
    """Angle of Arrival parameter class."""

    def __init__(
        self,
        gamma: float,
        d: float,
        range_bin_size: float,
        velocity_bin_size: float,
        doppler_fft_size: int,
        sidelobe_azi_db: float,
        sidelobe_ele_db: float,
        n_pc_preserve: int,
        azimuth_fov: Float[np.ndarray, "2"],
        elevation_fov: Float[np.ndarray, "2"],
        vec_angle: Float[np.ndarray, "aoa_fft_size"],
    ):
        self.gamma = gamma
        self.d = d
        self.range_bin_size = range_bin_size
        self.velocity_bin_size = velocity_bin_size
        self.doppler_fft_size = doppler_fft_size
        self.sidelobe_azi_db = sidelobe_azi_db
        self.sidelobe_ele_db = sidelobe_ele_db
        self.n_pc_preserve = n_pc_preserve
        self.azimuth_fov = azimuth_fov
        self.elevation_fov = elevation_fov
        self.vec_angle = vec_angle
