"""Angle FFT processing for MIMO radar data."""

from functools import cached_property
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Int

from ..dataset import RadarParam


class AngleFFT:
    """Angle FFT processing for MIMO radar data.

    Args:
        radar_param: Radar parameters containing antenna configuration.
        fft_size: Size of the FFT to be performed in (azimuth, elevation) dimensions
    """

    def __init__(self, radar_param: RadarParam, fft_size: tuple[int, int] = (256, 256)):
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
        """Rearange phasor on corresponding antenna position.

        Returns:
            A function that takes in a signal of shape (..., Rx, Tx) and returns a signal of shape (..., azi, ele).
        """

        def inner(
            signal: Complex[Array, "... Rx Tx"],
        ) -> Complex[np.ndarray, "... azi ele"]:
            """Inner function to rearange phasor using numpy.

            Args:
                signal: Input signal of shape (..., Rx, Tx).

            Returns:
                Rearanged signal of shape (..., azi, ele).
            """
            shape = list(signal.shape)
            shape[-2:] = self.antenna_layout  # type: ignore
            sig2D = np.zeros(shape, dtype=signal.dtype)
            sig2D[..., self.antenna_azi, self.antenna_ele] = signal
            return sig2D

        def pure_cb(
            signal: Complex[Array, "... Rx Tx"],
        ) -> Any:
            shape = list(signal.shape)
            shape[-2:] = self.antenna_layout  # type: ignore
            return jax.pure_callback(
                inner,
                jax.ShapeDtypeStruct(shape, signal.dtype),
                signal,
                vectorized=True,
            )

        return pure_cb

    def rearange_antenna(
        self, signal: Complex[Array, "Rx Tx"]
    ) -> Complex[Array, "azi ele"]:
        """Returns the signal rearanged on corresponding antenna position.

        !!! note
            Slower than np + pure_cb.

        Args:
            signal: Input signal of shape (Rx, Tx).

        Returns:
            Rearanged signal of shape (azi, ele).
        """
        sig2D = jnp.zeros(self.antenna_layout, dtype=signal.dtype)
        sig2D = sig2D.at[self.antenna_azi, self.antenna_ele].set(signal)
        return sig2D

    def phase_correction(
        self, signal: Complex[Array, "Rx Tx"], rd_indx: Int[Array, "2"]
    ) -> Complex[Array, "Rx Tx"]:
        """Doppler phase correction due to TDM MIMO.

        Args:
            signal: Input signal of shape (Rx, Tx).
            rd_indx: Range-Doppler index of shape (2, ).

        Returns:
            Phase corrected signal of shape (Rx, Tx).
        """
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
    ) -> tuple[Complex[Array, "azi"], Complex[Array, "azi ele"]]:
        """Perform angle FFT processing.

        Args:
            signal: Input signal of shape (Rx, Tx).
            rd_indx: Range-Doppler index of shape (2, ).

        Returns:
            1D azimuth FFT result of shape (azi).
            2D angle FFT result of shape (azi, ele).
        """
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
        """Perform 2D angle FFT processing.

        Args:
            signal: Input signal of shape (Rx, Tx).
            rd_indx: Range-Doppler index of shape (2, ).

        Returns:
            2D angle FFT result of shape (azi, ele).
        """
        phase_correct = self.phase_correction(signal, rd_indx)
        sig_antenna = self.rearange_phasor(phase_correct)
        sig2D = jnp.fft.fft2(sig_antenna, self.fft_size, norm="forward")
        sig2D = jnp.fft.fftshift(sig2D, axes=(-2, -1))
        return sig2D
