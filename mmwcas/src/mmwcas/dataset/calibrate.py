import numpy as np

from jaxtyping import Complex, Array
from scipy.io import loadmat
from .params import RadarParam

class CascadeCalibration:
    def __init__(
        self,
        calib_file: str,
        param: RadarParam,
        calibrationInterp: float = 5.0,
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
        self.calibrationInterp = calibrationInterp
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
            2 * np.pi * freq_calib / (param.numSamplePerChirp * calibrationInterp)
        )
        freq_calib = np.arange(param.numSamplePerChirp) * freq_calib[..., None]
        freq_calib = np.conj(np.exp(1j * freq_calib))
        self.freq_calib = np.transpose(freq_calib[..., None], (2, 3, 1, 0))

        # phase calibration
        phase_calib = self.PeakValMat[0, 0] / self.PeakValMat
        if not amp_clib:
            phase_calib /= np.abs(phase_calib)
        self.phase_calib = np.transpose(phase_calib)[None, None, ...]

    def __call__(self, signal: Complex[Array, "sample chirp Rx Tx"]):
        return signal * self.freq_calib * self.phase_calib
