import jax
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float, Complex
from .window import sym_hanning

class RangeFFT:
    def __init__(
        self, fft_size: int = 256, adc_samples: int = 256, norm: str = None
    ):
        self.fft_size = fft_size
        self.norm = norm
        self.window = sym_hanning(adc_samples)

    def __call__(
        self, signal_cube: Complex[Array, "samples chirp Rx Tx"]
    ) -> Complex[Array, "range chirp Rx Tx"]:

        # DC offset
        signal_cube = signal_cube - jnp.mean(signal_cube, axis=0)

        # windowing
        signal_cube = signal_cube * self.window[:, None, None, None]

        # FFT
        signal_cube = jnp.fft.fft(signal_cube, self.fft_size, axis=0, norm=self.norm)

        return signal_cube
