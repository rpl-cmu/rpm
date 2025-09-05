"""Doppler FFT processing module."""

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Complex

from .window import sym_hanning


class DopplerFFT:
    """Doppler FFT processing module.

    Args:
        fft_size: Size of the Doppler FFT.
        num_chirp: Number of chirps in the input signal cube.
        window: Whether to apply a Hanning window before FFT.
        norm: Normalization mode for the FFT.
    """

    def __init__(
        self,
        fft_size: int = 64,
        num_chirp: int = 64,
        window: bool = True,
        norm: Optional[str] = None,
    ):
        self.fft_size = fft_size
        self.norm = norm
        if window:
            self.window = sym_hanning(num_chirp)
        else:
            self.window = jnp.ones(num_chirp)

    def __call__(
        self, signal_cube: Complex[Array, "range chirp Rx Tx"]
    ) -> Complex[Array, "range doppler Rx Tx"]:
        """Applies Doppler FFT to the input signal cube.

        Args:
            signal_cube: Input signal cube of shape (range, chirp, Rx, Tx).

        Returns:
            Processed signal cube of shape (range, doppler, Rx, Tx).
        """
        # windowing
        signal_cube = signal_cube * self.window[None, :, None, None]

        # FFT
        signal_cube = jnp.fft.fft(signal_cube, self.fft_size, axis=1, norm=self.norm)
        signal_cube = jnp.fft.fftshift(signal_cube, axes=1)

        return signal_cube
