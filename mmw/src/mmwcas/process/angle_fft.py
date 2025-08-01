import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Int
from functools import cached_property
from typing import Tuple, Callable, Optional

from ..dataset import RadarParam


class AngleFFT:
    def __init__(self, radar_param: RadarParam, fft_size: Tuple[int, int] = (256, 256)):
        self.radar_param = radar_param
        self.antenna_azi = radar_param.antenna_azi
        self.antenna_ele = radar_param.antenna_ele
        self.antenna_layout = (
            np.max(self.antenna_azi) + 1,
            np.max(self.antenna_ele) + 1,
        )
        self.shape_rxtx = (radar_param.numRx, radar_param.numTx)
        self.dopplerFFTsize = radar_param.DopplerFFTSize
        self.numTx = radar_param.numTx
        self.fft_size = fft_size

    @cached_property
    def rearange_phasor(
        self,
    ) -> Callable[[Complex[Array, "... Rx Tx"]], Complex[Array, "... azi ele"]]:
        """rearange phasor on corresponding antenna position"""

        def __inner__(signal: Complex[Array, "... Rx Tx"]):
            shape = list(signal.shape)
            shape[-2:] = self.antenna_layout
            sig2D = np.zeros(shape, dtype=signal.dtype)
            sig2D[..., self.antenna_azi, self.antenna_ele] = signal
            return sig2D

        def pure_cb(
            signal: Complex[Array, "... Rx Tx"],
        ) -> Complex[Array, "... azi ele"]:
            shape = list(signal.shape)
            shape[-2:] = self.antenna_layout
            return jax.pure_callback(
                __inner__,
                jax.ShapeDtypeStruct(shape, signal.dtype),
                signal,
                vectorized=True,
            )

        return pure_cb

    def rearange_antenna(
        self, signal: Complex[Array, "Rx Tx"]
    ) -> Complex[Array, "azi ele"]:
        """slower than np + pure_cb"""
        sig2D = jnp.zeros(self.antenna_layout, dtype=signal.dtype)
        sig2D = sig2D.at[self.antenna_azi, self.antenna_ele].set(signal)
        return sig2D

    def phase_correction(
        self, signal: Complex[Array, "Rx Tx"], rd_indx: Int[Array, "2"]
    ) -> Complex[Array, "Rx Tx"]:
        """doppler phase correction due to TDM MIMO"""
        deltaPhi = (
            2
            * jnp.pi
            * (rd_indx[1] - self.dopplerFFTsize / 2)
            / (self.numTx * self.dopplerFFTsize)
        )
        correct_mat = jnp.exp(-1j * jnp.arange(self.numTx) * deltaPhi)
        return signal * correct_mat[None, :]

    def __call__(
        self, signal: Complex[Array, "Rx Tx"], rd_indx: Int[Array, "2"]
    ) -> Tuple[Complex[Array, "azi"], Complex[Array, "azi ele"]]:

        phase_correct = self.phase_correction(signal, rd_indx)
        sig_antenna = self.rearange_phasor(phase_correct)

        sig2D = jnp.fft.fftshift(
            jnp.fft.fft(sig_antenna, self.fft_size[0], axis=0), axes=0
        )
        sig_azi = sig2D[:, 0]
        sig_angle = jnp.fft.fftshift(
            jnp.fft.fft(sig2D, self.fft_size[1], axis=1), axes=1
        )

        return sig_azi, sig_angle

    def cube(
        self, signal: Complex[Array, "Rx Tx"], rd_indx: Int[Array, "2"]
    ) -> Complex[Array, "azi ele"]:
        phase_correct = self.phase_correction(signal, rd_indx)
        sig_antenna = self.rearange_phasor(phase_correct)
        sig2D = jnp.fft.fft2(sig_antenna, self.fft_size, norm="forward")
        sig2D = jnp.fft.fftshift(sig2D, axes=(-2, -1))
        return sig2D
