"""Range FFT processing module."""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float

from .window import sym_hanning


class RangeFFT:
    """Range FFT processing module.

    Args:
        fft_size: Size of the FFT to compute.
        adc_samples: Number of ADC samples in the input signal.
        norm: Normalization mode for the FFT. Can be 'forward', 'backward',
    """

    def __init__(
        self, fft_size: int = 256, adc_samples: int = 256, norm: Optional[str] = None
    ):
        self.fft_size = fft_size
        self.norm = norm
        self.window = sym_hanning(adc_samples)

    def __call__(
        self, signal_cube: Complex[Array, "samples chirp Rx Tx"]
    ) -> Complex[Array, "range chirp Rx Tx"]:
        """Applies Range FFT processing to the input signal cube.

        Args:
            signal_cube: Input signal cube of shape (samples, chirp, Rx, Tx).

        Returns:
            Processed signal cube of shape (range, chirp, Rx, Tx).
        """
        # DC offset
        signal_cube = signal_cube - jnp.mean(signal_cube, axis=0)

        # windowing
        signal_cube = signal_cube * self.window[:, None, None, None]

        # FFT
        signal_cube = jnp.fft.fft(signal_cube, self.fft_size, axis=0, norm=self.norm)

        return signal_cube
