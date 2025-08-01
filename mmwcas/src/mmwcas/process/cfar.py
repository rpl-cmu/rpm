import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import correlate2d, correlate
from jaxtyping import Array, Complex, Float


class CFAR:
    def __init__(
        self,
        rangeFFTSize: int,
        DopplerFFTSize: int,
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
        self.preserve_mask = np.zeros((rangeFFTSize, DopplerFFTSize), dtype=bool)
        self.preserve_mask[discard_range_close:-discard_range_far, :] = True
        self.preserve_mask = jnp.asarray(self.preserve_mask)

        self.pad_r = train_range + guard_range
        self.pad_d = train_doppler + guard_doppler
        w_range = 2 * (self.pad_r) + 1
        w_doppler = 2 * (self.pad_d) + 1
        self.window = np.ones((w_range, w_doppler), dtype=np.float32)
        self.window[train_range:-train_range, train_doppler:-train_doppler] = 0
        self.window /= self.window.sum()
        self.window = jnp.asarray(self.window)

    def __call__(self, signal_cube: Complex[Array, "range doppler Rx Tx"]):

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


class CFAR_CASO:
    def __init__(
        self,
        train_range: int = 8,
        guard_range: int = 8,
        train_doppler: int = 4,
        guard_doppler: int = 0,
        K0_range: float = 5.0,
        K0_doppler: float = 3.0,
        discard_range_close: int = 10,
        discard_range_far: int = 20,
    ):
        self.pad_r = train_range + guard_range
        self.pad_d = train_doppler + guard_doppler

        # discard detect object around DC
        self.discard_close, self.discard_far = discard_range_close, discard_range_far

        # caso
        r_ker = np.zeros((2 * self.pad_r + 1), dtype=np.float32)
        self.r_ker_a, self.r_ker_b = r_ker.copy(), r_ker.copy()
        self.r_ker_a[:train_range], self.r_ker_b[-train_range:] = 1, 1
        self.r_ker_a /= self.r_ker_a.sum()
        self.r_ker_b /= self.r_ker_b.sum()

        d_ker = np.zeros((2 * self.pad_d + 1), dtype=np.float32)
        self.d_ker_a, self.d_ker_b = d_ker.copy(), d_ker.copy()
        self.d_ker_a[:train_doppler], self.d_ker_b[-train_doppler:] = 1, 1
        self.d_ker_a /= self.d_ker_a.sum()
        self.d_ker_b /= self.d_ker_b.sum()

        self.k0_r, self.k0_d = K0_range, K0_doppler

    def caso(
        self,
        signal: Float[Array, "n"],
        ker_a: Float[Array, "w"],
        ker_b: Float[Array, "w"],
        k0: float,
        pad: int,
    ):
        """
        1D CFAR CASO
        """
        cor_a = correlate(signal, ker_a, mode="valid")
        cor_b = correlate(signal, ker_b, mode="valid")
        noise = jnp.minimum(cor_a, cor_b)
        detect = signal[pad:-pad] > k0 * noise
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
            sig_pad_r, self.r_ker_a, self.r_ker_b, self.k0_r, self.pad_r
        )
        detect_r, noise = detect_r.swapaxes(0, 1), noise.swapaxes(0, 1)
        detect_r = jnp.pad(detect_r, ((self.discard_close, self.discard_far), (0, 0)))
        noise = jnp.pad(
            noise, ((self.discard_close, self.discard_far), (0, 0)), constant_values=1
        )
        detect_d, _ = jax.vmap(self.caso, in_axes=(0, None, None, None, None))(
            sig_pad_d, self.d_ker_a, self.d_ker_b, self.k0_d, self.pad_d
        )

        snr = signal / noise
        obj_mask = jnp.logical_and(detect_r, detect_d)

        return obj_mask, signal, snr
