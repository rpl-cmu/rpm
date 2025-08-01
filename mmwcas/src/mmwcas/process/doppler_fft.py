import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Complex
from .window import sym_hanning


class DopplerFFT:
    def __init__(
        self,
        fft_size: int = 64,
        num_chirp: int = 64,
        window: Bool = True,
        norm: str = None,
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

        # windowing
        signal_cube = signal_cube * self.window[None, :, None, None]

        # FFT
        signal_cube = jnp.fft.fft(signal_cube, self.fft_size, axis=1, norm=self.norm)
        signal_cube = jnp.fft.fftshift(signal_cube, axes=1)

        return signal_cube
