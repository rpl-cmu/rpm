import numpy as np
import numba as nb
from typing import Tuple
from jaxtyping import Float, Array

spec = [
    ("gamma", nb.float32),
    ("d", nb.float32),
    ("rangeBinSize", nb.float32),
    ("velocityBinSize", nb.float32),
    ("dopplerFFTSize", nb.int32),
    ("sidelobe_azi_dB", nb.float32),
    ("sidelobe_ele_dB", nb.float32),
    ("n_pc_preserve", nb.int32),
    ("azimuth_fov", nb.float32[:]),
    ("elevation_fov", nb.float32[:]),
    ("vec_angle", nb.float32[:]),
]


@nb.experimental.jitclass(spec)
class AoAparam:
    def __init__(
        self,
        gamma: float,
        d: float,
        rangeBinSize: float,
        velocityBinSize: float,
        dopplerFFTSize: int,
        sidelobe_azi_dB: float,
        sidelobe_ele_dB: float,
        n_pc_preserve: int,
        azimuth_fov: Float[np.ndarray, "2"],
        elevation_fov: Float[np.ndarray, "2"],
        vec_angle: Float[np.ndarray, "aoa_fft_size"],
    ):
        self.gamma = gamma
        self.d = d
        self.rangeBinSize = rangeBinSize
        self.velocityBinSize = velocityBinSize
        self.dopplerFFTSize = dopplerFFTSize
        self.sidelobe_azi_dB = sidelobe_azi_dB
        self.sidelobe_ele_dB = sidelobe_ele_dB
        self.n_pc_preserve = n_pc_preserve
        self.azimuth_fov = azimuth_fov
        self.elevation_fov = elevation_fov
        self.vec_angle = vec_angle
