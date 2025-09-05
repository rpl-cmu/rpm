"""Azimuth FFT processing module."""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex

from ..dataset import RadarParam


class AzimuthFFT:
    """Azimuth FFT processing class.

    Args:
        param: Radar parameters.
        angele_fft_size: Size of the angle FFT.
        range_bin_min: Minimum range bin index to consider.
        range_bin_max: Maximum range bin index to consider.
        angle_bin_min: Minimum angle bin index to consider.
        angle_bin_max: Maximum angle bin index to consider.
    """

    def __init__(
        self,
        param: RadarParam,
        angele_fft_size: int = 256,
        range_bin_min: int = 5,
        range_bin_max: int = 236,
        angle_bin_min: int = 14,
        angle_bin_max: int = 242,
    ) -> None:
        self.param = param
        self.angle_fft_size = angele_fft_size
        self.range_bin_min = range_bin_min
        self.range_bin_max = range_bin_max
        self.angle_bin_min = angle_bin_min
        self.angle_bin_max = angle_bin_max

        antenna_azi = np.array(param.antenna_azi).reshape(-1)
        antenna_ele = np.array(param.antenna_ele).reshape(-1)
        self.azi_element = antenna_ele == 0
        _, self.azi_unique = np.unique(antenna_azi[self.azi_element], return_index=True)
        self.window = jnp.hanning(self.azi_unique.size)

        theta = (
            np.linspace(-angele_fft_size // 2, angele_fft_size // 2, angele_fft_size)
            / angele_fft_size
        )[angle_bin_min:angle_bin_max]
        sin = -2 * theta
        cos = np.sqrt(1 - sin**2)
        bins = np.linspace(
            range_bin_min, range_bin_max, range_bin_max - range_bin_min, endpoint=True
        )
        ranges = bins * param.rangeResolution
        r_, sin_ = np.meshgrid(ranges, sin, indexing="ij")
        _, cos_ = np.meshgrid(ranges, cos, indexing="ij")
        self.x_axis, self.y_axis = r_ * cos_, r_ * sin_

    def __call__(
        self, signal: Complex[Array, "range doppler Rx Tx"]
    ) -> Complex[Array, "range_trimmed doppler azimuth"]:
        s_r, s_d, s_rx, s_tx = signal.shape
        signal = signal.reshape(s_r, s_d, -1)
        signal = signal[:, :, self.azi_element]
        signal = signal[:, :, self.azi_unique]
        signal = signal * self.window[None, None, :]

        signal = jnp.fft.fft(signal, n=self.angle_fft_size, axis=-1, norm="forward")
        signal = jnp.fft.fftshift(signal, axes=-1)

        signal = signal[self.range_bin_min : self.range_bin_max]
        signal = signal[:, :, self.angle_bin_min : self.angle_bin_max]

        return signal
