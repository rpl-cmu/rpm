"""Pipeline processing modules for radar signal processing."""

from functools import cached_property
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import correlate2d
from jaxtyping import Array, Bool, Complex, Float

from ..dataset.params import RadarParam
from .angle_fft import AngleFFT
from .angle_of_arrival import AOAPC
from .azimuth_fft import AzimuthFFT
from .cfar import CFAR, CFARCASO
from .doppler_fft import DopplerFFT
from .range_fft import RangeFFT


class RangeProc:
    """Range processing module.

    Args:
        radar_param: Radar parameters.
        normalize_factor: Normalization factor for output range image.
    """

    def __init__(self, radar_param: RadarParam, normalize_factor=5e4):
        self.range_proc = RangeFFT(
            fft_size=radar_param.rangeFFTSize, adc_samples=radar_param.numADCSample
        )
        self.normalize_factor = normalize_factor

    def __call__(
        self, sig_tensor: Complex[Array, "sample chirp rx tx"]
    ) -> Float[Array, "range chirp"]:
        """Process the input signal tensor to generate range image.

        Args:
            sig_tensor: Input signal tensor of shape (sample, chirp, rx, tx).

        Returns:
            Range image of shape (range, ...).
        """
        sig_tensor = self.range_proc(sig_tensor)
        r, d, rx, tx = sig_tensor.shape
        sig_tensor = sig_tensor.reshape(r, d, -1)
        range_img = jnp.mean(jnp.abs(sig_tensor), axis=-1)

        range_img = jnp.clip(range_img / self.normalize_factor, 0, 1)

        return range_img


class RangeAzimuthProc:
    """Range-Azimuth processing module.

    Args:
        radar_param: Radar parameters.
        angele_fft_size: Angle FFT size.
        range_bin_min: Minimum range bin to consider.
        range_bin_max: Maximum range bin to consider.
        angle_3db: 3dB angle for azimuth processing.
        normalize_factor: Normalization factor for output range-azimuth image.
        output_normalize: Whether to normalize the output range-azimuth image.
    """

    def __init__(
        self,
        radar_param: RadarParam,
        angele_fft_size=256,
        range_bin_min=5,
        range_bin_max=236,
        angle_3db=85.0,
        normalize_factor=10,
        output_normalize=True,
    ):
        self.range_proc = RangeFFT(
            fft_size=radar_param.rangeFFTSize,
            adc_samples=radar_param.numADCSample,
            norm="forward",
        )

        angle_bin_min = int((90 - angle_3db) / 90 * (angele_fft_size // 2))
        angle_bin_max = angele_fft_size - angle_bin_min
        self.azimuth_proc = AzimuthFFT(
            param=radar_param,
            angele_fft_size=angele_fft_size,
            range_bin_min=range_bin_min,
            range_bin_max=range_bin_max,
            angle_bin_min=angle_bin_min,
            angle_bin_max=angle_bin_max,
        )
        self.normalize_factor = normalize_factor
        self.x_axis = self.azimuth_proc.x_axis
        self.y_axis = self.azimuth_proc.y_axis

        self.output_normalize = output_normalize

    def __call__(
        self, sig_tensor: Complex[Array, "sample chirp rx tx"]
    ) -> Float[Array, "range azimuth"]:
        """Process the input signal tensor to generate range-azimuth image.

        Args:
            sig_tensor: Input signal tensor of shape (sample, chirp, rx, tx).

        Returns:
            Range-azimuth image of shape (range, azimuth).
        """
        sig_tensor = self.range_proc(sig_tensor)
        # sig_tensor = self.doppler_proc(sig_tensor)
        sig_tensor = self.azimuth_proc(sig_tensor)

        range_azimuth = jnp.mean(jnp.abs(sig_tensor), axis=1)

        if self.output_normalize:
            return jnp.clip(range_azimuth / self.normalize_factor, 0, 1)

        return range_azimuth


class RangeAzimuthCFARProc:
    """Range-Azimuth processing module with CFAR detection.

    Args:
        radar_param: Radar parameters.
        angele_fft_size: Angle FFT size.
        range_bin_min: Minimum range bin to consider.
        range_bin_max: Maximum range bin to consider.
        normalize_factor: Normalization factor for output range-azimuth image.
        thresh_snr: SNR threshold for CFAR detection.
        train_range: Number of training cells in range dimension.
        guard_range: Number of guard cells in range dimension.
        train_azimuth: Number of training cells in azimuth dimension.
        guard_azimuth: Number of guard cells in azimuth dimension.
    """

    def __init__(
        self,
        radar_param: RadarParam,
        angele_fft_size: int = 256,
        range_bin_min: int = 5,
        range_bin_max: int = 236,
        normalize_factor: float = 1e6,
        thresh_snr: float = 5,
        train_range: int = 8,
        guard_range: int = 8,
        train_azimuth: int = 8,
        guard_azimuth: int = 0,
    ):
        self.range_proc = RangeFFT(
            fft_size=radar_param.rangeFFTSize, adc_samples=radar_param.numADCSample
        )
        self.azimuth_proc = AzimuthFFT(
            param=radar_param,
            angele_fft_size=angele_fft_size,
            range_bin_min=range_bin_min,
            range_bin_max=range_bin_max,
        )
        self.normalize_factor = normalize_factor
        self.x_axis = self.azimuth_proc.x_axis
        self.y_axis = self.azimuth_proc.y_axis

        self.thresh_snr = thresh_snr

        # discard positive and negative frequencies around DC
        self.preserve_mask = np.zeros(
            (radar_param.rangeFFTSize, angele_fft_size), dtype=bool
        )
        # self.preserve_mask[discard_range_close:-discard_range_far, :] = True
        self.preserve_mask = self.preserve_mask[range_bin_min:range_bin_max]
        self.preserve_mask = True
        self.preserve_mask = jnp.asarray(self.preserve_mask)

        self.pad_r = train_range + guard_range
        self.pad_a = train_azimuth + guard_azimuth
        w_range = 2 * (self.pad_r) + 1
        w_azimuth = 2 * (self.pad_a) + 1
        self.window = np.ones((w_range, w_azimuth), dtype=np.float32)
        self.window[train_range:-train_range, train_azimuth:-train_azimuth] = 0
        self.window /= self.window.sum()
        self.window = jnp.asarray(self.window)

    def __call__(
        self, sig_tensor: Complex[Array, "sample chirp rx tx"]
    ) -> tuple[Bool[Array, "range azimuth"], Float[Array, "range azimuth"]]:
        """Process the input signal tensor to generate range-azimuth image and CFAR mask.

        Args:
            sig_tensor: Input signal tensor of shape (sample, chirp, rx, tx).

        Returns:
            CFAR detection mask of shape (range, azimuth).
            Range-azimuth image of shape (range, azimuth).
        """
        sig_tensor = self.range_proc(sig_tensor)
        # sig_tensor = self.doppler_proc(sig_tensor)
        sig_tensor = self.azimuth_proc(sig_tensor)

        range_azimuth = jnp.mean(jnp.abs(sig_tensor), axis=1)

        radar_img = jnp.clip(range_azimuth / self.normalize_factor, 0, 1)

        sig_pad = jnp.pad(
            radar_img,
            ((self.pad_r, self.pad_r), (self.pad_a, self.pad_a)),
            mode="wrap",
        )
        noise = correlate2d(sig_pad, self.window, mode="valid")  # type: ignore
        snr = radar_img / noise

        obj_mask = jnp.logical_and(snr > self.thresh_snr, self.preserve_mask)

        return obj_mask, radar_img


class RangeDopplerCFARProc:
    """Range-Doppler processing module with CFAR detection.

    Args:
        radar_param: Radar parameters.
        thresh_snr: SNR threshold for CFAR detection.
        train_range: Number of training cells in range dimension.
        guard_range: Number of guard cells in range dimension.
        train_doppler: Number of training cells in doppler dimension.
        guard_doppler: Number of guard cells in doppler dimension.
        discard_range_close: Number of range bins to discard from close range.
        discard_range_far: Number of range bins to discard from far range.
        normalize_factor: Normalization factor for output range-doppler image.
    """

    def __init__(
        self,
        radar_param: RadarParam,
        thresh_snr=3,
        train_range=8,
        guard_range=8,
        train_doppler=4,
        guard_doppler=0,
        discard_range_close=10,
        discard_range_far=20,
        normalize_factor=10,
    ):
        self.range_proc = RangeFFT(
            fft_size=radar_param.rangeFFTSize,
            adc_samples=radar_param.numADCSample,
            norm="forward",
        )
        self.doppler_proc = DopplerFFT(
            fft_size=radar_param.DopplerFFTSize,
            num_chirp=radar_param.numChirp,
            norm="forward",
        )
        self.cfar = CFAR(
            range_fft_size=radar_param.rangeFFTSize,
            doppler_fft_size=radar_param.DopplerFFTSize,
            thresh_snr=thresh_snr,
            train_range=train_range,
            guard_range=guard_range,
            train_doppler=train_doppler,
            guard_doppler=guard_doppler,
            discard_range_close=discard_range_close,
            discard_range_far=discard_range_far,
        )

        self.normalize_factor = normalize_factor

    def __call__(
        self, sig_tensor: Complex[Array, "sample chirp rx tx"]
    ) -> tuple[Bool[Array, "range doppler"], Float[Array, "range doppler"]]:
        """Process the input signal tensor to generate range-doppler image and CFAR mask.

        Args:
            sig_tensor: Input signal tensor of shape (sample, chirp, rx, tx).

        Returns:
            CFAR detection mask of shape (range, doppler).
            Range-doppler image of shape (range, doppler).
        """
        sig_tensor = self.range_proc(sig_tensor)
        sig_tensor = self.doppler_proc(sig_tensor)

        obj_mask, rd_image, snr = self.cfar(sig_tensor)
        rd_image = jnp.clip(rd_image / self.normalize_factor, 0, 1)

        return obj_mask, rd_image


class RadarCubeProc:
    """Radar cube processing module.

    Args:
        radar_param: Radar parameters.
        angle_fft_size: Angle FFT size.

    """

    def __init__(
        self, radar_param: RadarParam, angle_fft_size: Optional[tuple[int, int]] = None
    ):
        self.range_proc = RangeFFT(
            fft_size=radar_param.rangeFFTSize,
            adc_samples=radar_param.numADCSample,
            norm="forward",
        )
        self.doppler_proc = DopplerFFT(
            fft_size=radar_param.DopplerFFTSize,
            num_chirp=radar_param.numChirp,
            window=False,
            norm="forward",
        )

        if angle_fft_size is None:
            angle_fft_size = (
                int(np.max(radar_param.antenna_azi) + 1),
                int(np.max(radar_param.antenna_ele) + 1),
            )

        self.afft = AngleFFT(radar_param=radar_param, fft_size=angle_fft_size)
        self.afft_proc = jax.vmap(jax.vmap(self.afft.cube))

        self.rd_all = jnp.array(
            [
                [r, d]
                for r in range(radar_param.rangeFFTSize)
                for d in range(radar_param.DopplerFFTSize)
            ]
        ).reshape(radar_param.rangeFFTSize, radar_param.DopplerFFTSize, 2)

    def __call__(
        self, sig_tensor: Complex[Array, "sample chirp rx tx"]
    ) -> Complex[Array, "range doppler azi ele"]:
        """Process the input signal tensor to generate radar cube.

        Args:
            sig_tensor: Input signal tensor of shape (sample, chirp, rx, tx).

        Returns:
            Radar cube of shape (range, doppler, azimuth, elevation).
        """
        sig_tensor = self.range_proc(sig_tensor)
        sig_tensor = self.doppler_proc(sig_tensor)
        sig_tensor = self.afft_proc(sig_tensor, self.rd_all)
        return sig_tensor


class RadarPCProc:
    """Radar point cloud processing module.

    Args:
        radar_param: Radar parameters.
        train_range: Number of training cells in range dimension for CFAR.
        guard_range: Number of guard cells in range dimension for CFAR.
        train_doppler: Number of training cells in doppler dimension for CFAR.
        guard_doppler: Number of guard cells in doppler dimension for CFAR.
        snr_range: SNR threshold in range dimension for CFAR.
        snr_doppler: SNR threshold in doppler dimension for CFAR.
        discard_range_close: Number of range bins to discard from close range for CFAR.
        discard_range_far: Number of range bins to discard from far range for CFAR.
        angle_fft_size: Angle FFT size.
        azimuth_fov: Azimuth field of view.
        elevation_fov: Elevation field of view.
        aoa_gamma_db: Gamma value in dB for AOAPC.
        sidelobe_azi_db: Sidelobe level in dB for azimuth.
        sidelobe_ele_db: Sidelobe level in dB for elevation.
    """

    def __init__(
        self,
        radar_param: RadarParam,
        train_range: int = 8,
        guard_range: int = 8,
        train_doppler: int = 4,
        guard_doppler: int = 0,
        snr_range: float = 5.0,
        snr_doppler: float = 3.0,
        discard_range_close: int = 10,
        discard_range_far: int = 20,
        angle_fft_size: int = 256,
        azimuth_fov: Tuple[float, float] = (-80.0, 80.0),
        elevation_fov: Tuple[float, float] = (-20.0, 20.0),
        aoa_gamma_db: float = 0.2,
        sidelobe_azi_db: float = 1.0,
        sidelobe_ele_db: float = 0.0,
    ):
        self.range_proc = RangeFFT(
            fft_size=radar_param.rangeFFTSize,
            adc_samples=radar_param.numADCSample,
            norm="forward",
        )
        self.doppler_proc = DopplerFFT(
            fft_size=radar_param.DopplerFFTSize,
            num_chirp=radar_param.numChirp,
            window=False,
            norm="forward",
        )
        self.cfar = CFARCASO(
            train_range=train_range,
            guard_range=guard_range,
            train_doppler=train_doppler,
            guard_doppler=guard_doppler,
            snr_range=snr_range,
            snr_doppler=snr_doppler,
            discard_range_close=discard_range_close,
            discard_range_far=discard_range_far,
        )
        self.angle_fft_size = angle_fft_size
        afft = AngleFFT(
            radar_param=radar_param, fft_size=(angle_fft_size, angle_fft_size)
        )
        self.angle_fft = jax.jit(jax.vmap(afft))

        self.aoa_pc = AOAPC(
            radar_param=radar_param,
            gamma_db=aoa_gamma_db,
            sidelobe_azi_db=sidelobe_azi_db,
            sidelobe_ele_db=sidelobe_ele_db,
            azimuth_fov=azimuth_fov,
            elevation_fov=elevation_fov,
            aoa_fft_size=angle_fft_size,
        )

    @cached_property
    def fft_procs(self) -> Callable:
        @jax.jit
        def inner(
            sig_tensor: Complex[Array, "sample chirp rx tx"],
        ) -> tuple[
            Bool[Array, "range doppler"],
            Complex[Array, "range doppler rx tx"],
            Float[Array, "range doppler"],
        ]:
            rfft = self.range_proc(sig_tensor)
            rdfft = self.doppler_proc(rfft)
            obj_mask, _, snr = self.cfar(rdfft)
            return obj_mask, rdfft, snr

        return inner

    def __call__(
        self, sig_tensor: Complex[Array, "sample chirp rx tx"]
    ) -> Float[np.ndarray, "#points 3"]:
        obj_mask, rdfft, snrs = self.fft_procs(sig_tensor)
        sig_pick = np.asarray(rdfft)[obj_mask]
        objs = np.argwhere(np.asarray(obj_mask))
        snrs = np.asarray(snrs)[obj_mask]

        sig_azis, sig_angles = self.angle_fft(sig_pick, objs)
        pc = self.aoa_pc(np.asarray(sig_azis), np.asarray(sig_angles), objs, snrs)

        return np.asarray(pc)
