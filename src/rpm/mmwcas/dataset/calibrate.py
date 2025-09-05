"""Calibration module for cascade radar systems."""

import numpy as np
from jaxtyping import Complex64
from scipy.io import loadmat

from .params import RadarParam


class CascadeCalibration:
    """Calibration class for cascade radar systems.

    Args:
        calib_file: Path to the calibration .mat file.
        param: Radar parameters.
        calibration_interp: Interpolation factor for calibration.
        amp_clib: Whether to apply amplitude calibration.

    """

    def __init__(
        self,
        calib_file: str,
        param: RadarParam,
        calibration_interp: float = 5.0,
        amp_clib: bool = False,
    ):
        mat = loadmat(calib_file)
        self.calib_slope, self.calib_sampling_rate = mat["params"][0, 0]
        self.calib_slope = self.calib_slope[0, 0] * 1e12
        self.calib_sampling_rate = self.calib_sampling_rate[0, 0]
        (
            self.angleMat,
            self.RangeMat,
            self.PeakValMat,
            self.RxMismatch,
            self.TxMismatch,
            self.Rx_fft,
        ) = mat["calibResult"][0, 0]

        self.param = param
        self.tx_order = param.TxOrder

        self.PeakValMat = self.PeakValMat[self.tx_order]
        self.RangeMat = self.RangeMat[self.tx_order].astype(np.int16)

        # frequency calibration
        freq_calib = (
            (self.RangeMat - self.RangeMat[0, 0])
            * self.calib_sampling_rate
            / param.adcSampleRate
            * param.chirpSlope
            / self.calib_slope
        )
        freq_calib = (
            2 * np.pi * freq_calib / (param.numSamplePerChirp * calibration_interp)
        )
        freq_calib = np.arange(param.numSamplePerChirp) * freq_calib[..., None]
        freq_calib = np.conj(np.exp(1j * freq_calib))
        self.freq_calib = np.transpose(freq_calib[..., None], (2, 3, 1, 0))
        self.freq_calib = self.freq_calib.astype(np.complex64)

        # phase calibration
        phase_calib = self.PeakValMat[0, 0] / self.PeakValMat
        if not amp_clib:
            phase_calib /= np.abs(phase_calib)
        self.phase_calib = np.transpose(phase_calib)[None, None, ...]
        self.phase_calib = self.phase_calib.astype(np.complex64)

    def __call__(
        self, signal: Complex64[np.ndarray, "sample chirp Rx Tx"]
    ) -> Complex64[np.ndarray, "sample chirp Rx Tx"]:
        return signal * self.freq_calib * self.phase_calib
