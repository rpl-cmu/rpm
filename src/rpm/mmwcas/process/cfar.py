"""CFAR processing module."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import correlate, correlate2d
from jaxtyping import Array, Bool, Complex, Float


class CFAR:
    """2D CA-CFAR implementation.

    Args:
        rangeFFTSize: Size of the range FFT.
        DopplerFFTSize: Size of the Doppler FFT.
        thresh_snr: SNR threshold for detection.
        train_range: Number of training cells in range dimension.
        guard_range: Number of guard cells in range dimension.
        train_doppler: Number of training cells in Doppler dimension.
        guard_doppler: Number of guard cells in Doppler dimension.
        discard_range_close: Number of range bins to discard close to DC.
        discard_range_far: Number of range bins to discard far from DC.
    """

    def __init__(
        self,
        range_fft_size: int,
        doppler_fft_size: int,
        thresh_snr: float = 2,
        train_range: int = 8,
        guard_range: int = 8,
        train_doppler: int = 4,
        guard_doppler: int = 0,
        discard_range_close: int = 10,
        discard_range_far: int = 20,
    ):
        self.thresh_snr = thresh_snr

        # discard positive and negative frequencies around DC
        self.preserve_mask = np.zeros((range_fft_size, doppler_fft_size), dtype=bool)
        self.preserve_mask[discard_range_close:-discard_range_far, :] = True
        self.preserve_mask = jnp.asarray(self.preserve_mask)

        self.pad_r = train_range + guard_range
        self.pad_d = train_doppler + guard_doppler
        w_range = 2 * (self.pad_r) + 1
        w_doppler = 2 * (self.pad_d) + 1
        window = np.ones((w_range, w_doppler), dtype=np.float32)
        window[train_range:-train_range, train_doppler:-train_doppler] = 0
        window /= window.sum()
        self.window = jnp.asarray(window)

    def __call__(
        self, signal_cube: Complex[Array, "range doppler Rx Tx"]
    ) -> tuple[
        Bool[Array, "range doppler"],
        Float[Array, "range doppler"],
        Float[Array, "range doppler"],
    ]:
        s_r, s_d, _, _ = signal_cube.shape
        range_dopp = signal_cube.reshape(s_r, s_d, -1)

        # non-coherent signal combination along the antenna array
        signal = jnp.mean(jnp.abs(range_dopp), axis=-1)
        sig_pad = jnp.pad(
            signal,
            ((self.pad_r, self.pad_r), (self.pad_d, self.pad_d)),
            mode="wrap",
        )
        noise = correlate2d(sig_pad, self.window, mode="valid")
        snr = signal / noise

        obj_mask = jnp.logical_and(snr > self.thresh_snr, self.preserve_mask)

        return obj_mask, signal, snr


class CFARCASO:
    """2D CFAR CASO implementation.

    Args:
        train_range: Number of training cells in range dimension.
        guard_range: Number of guard cells in range dimension.
        train_doppler: Number of training cells in Doppler dimension.
        guard_doppler: Number of guard cells in Doppler dimension.
        K0_range: Scaling factor in range dimension.
        K0_doppler: Scaling factor in Doppler dimension.
        discard_range_close: Number of range bins to discard close to DC.
        discard_range_far: Number of range bins to discard far from DC.
    """

    def __init__(
        self,
        train_range: int = 8,
        guard_range: int = 8,
        train_doppler: int = 4,
        guard_doppler: int = 0,
        snr_range: float = 5.0,
        snr_doppler: float = 3.0,
        discard_range_close: int = 10,
        discard_range_far: int = 20,
    ):
        self.pad_r = train_range + guard_range
        self.pad_d = train_doppler + guard_doppler

        # discard detect object around DC
        self.discard_close, self.discard_far = discard_range_close, discard_range_far

        # caso
        def make_caso_kernels(train, pad):
            ker = np.zeros((2 * pad + 1), dtype=np.float32)
            ker_a, ker_b = ker.copy(), ker.copy()
            ker_a[:train], ker_b[-train:] = 1, 1
            ker_a /= ker_a.sum()
            ker_b /= ker_b.sum()
            return jnp.asarray(ker_a), jnp.asarray(ker_b)

        self.r_ker_a, self.r_ker_b = make_caso_kernels(train_range, self.pad_r)
        self.d_ker_a, self.d_ker_b = make_caso_kernels(train_doppler, self.pad_d)

        self.snr_r, self.snr_d = snr_range, snr_doppler

    def caso(
        self,
        signal: Float[Array, "n"],
        ker_a: Float[Array, "w"],
        ker_b: Float[Array, "w"],
        snr: float,
        pad: int,
    ):
        """1D CFAR CASO.

        Args:
            signal: 1D signal array.
            ker_a: Kernel A for correlation.
            ker_b: Kernel B for correlation.
            snr: SNR threshold.
            pad: Padding size.
        """
        cor_a = correlate(signal, ker_a, mode="valid")
        cor_b = correlate(signal, ker_b, mode="valid")
        noise = jnp.minimum(cor_a, cor_b)
        detect = signal[pad:-pad] > snr * noise
        return detect, noise

    def __call__(self, signal_cube: Complex[Array, "range doppler Rx Tx"]):
        s_r, s_d, _, _ = signal_cube.shape
        range_dopp = signal_cube.reshape(s_r, s_d, -1)

        # non-coherent signal combination along the antenna array
        signal = jnp.sum(jnp.abs(range_dopp) ** 2, axis=-1) + 1
        sig_discard = signal[self.discard_close : -self.discard_far]
        sig_pad_r = jnp.concat(
            (sig_discard[: self.pad_r], sig_discard, sig_discard[-self.pad_r :]), axis=0
        )
        sig_pad_d = jnp.pad(signal, ((0, 0), (self.pad_d, self.pad_d)), mode="wrap")

        # detection
        detect_r, noise = jax.vmap(self.caso, in_axes=(1, None, None, None, None))(
            sig_pad_r, self.r_ker_a, self.r_ker_b, self.snr_r, self.pad_r
        )
        detect_r, noise = detect_r.swapaxes(0, 1), noise.swapaxes(0, 1)
        detect_r = jnp.pad(detect_r, ((self.discard_close, self.discard_far), (0, 0)))
        noise = jnp.pad(
            noise, ((self.discard_close, self.discard_far), (0, 0)), constant_values=1
        )
        detect_d, _ = jax.vmap(self.caso, in_axes=(0, None, None, None, None))(
            sig_pad_d, self.d_ker_a, self.d_ker_b, self.snr_d, self.pad_d
        )

        snr = signal / noise
        obj_mask = jnp.logical_and(detect_r, detect_d)

        return obj_mask, signal, snr
