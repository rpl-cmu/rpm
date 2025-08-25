"""Angle of Arrival Point Cloud Processing Module."""

from typing import Tuple

import numba as nb
import numpy as np
from jaxtyping import Bool, Complex, Float, Int

from ..dataset import RadarParam
from .aoa_param import AoAparam

nb.config.THREADING_LAYER = "omp"  # type: ignore


@nb.njit
def peak_detect(
    signal: Complex[np.ndarray, "n"], gamma: float, sidelobe_db: float
) -> Int[np.ndarray, "m 1"]:
    """Peak detection algorithm for 1D signal.

    Args:
        signal: 1D complex input signal.
        gamma: Threshold factor for peak detection.
        sidelobe_db: Sidelobe threshold in dB.

    Returns:
        Indices of detected peaks in the signal.
    """
    N = signal.shape[0]
    spectrum = np.abs(signal)

    minVal, maxVal, maxLoc, extendLoc = np.inf, 0, 0, 0
    locateMax, initStage = False, True
    max_locs = np.zeros(N, dtype=np.bool_)

    i = 0
    while i < (N + extendLoc - 1):
        i_loc = i % N
        currentVal = spectrum[i_loc]

        # record the current max value and location
        if currentVal > maxVal:
            maxVal = currentVal
            maxLoc = i_loc

        # record for the current min value and location
        if currentVal < minVal:
            minVal = currentVal

        if locateMax:
            if currentVal < maxVal / gamma:
                max_locs[maxLoc] = True
                minVal = currentVal
                locateMax = 0
        else:
            if currentVal > minVal * gamma:
                locateMax = True
                maxVal = currentVal
                if initStage == True:
                    extendLoc = i
                    initStage = False
        i = i + 1

    sideLobeThresh = np.max(spectrum) * (10 ** (-sidelobe_db / 10))
    pkea_mask = np.logical_and(spectrum >= sideLobeThresh, max_locs)
    return np.argwhere(pkea_mask)


@nb.njit
def to_xyzsv(
    rd_indx: Int[np.ndarray, "2"] | list,
    aoa_indx: Int[np.ndarray, "2"] | list,
    snr: Float[np.ndarray, "1"],
    p: AoAparam,
) -> Float[np.ndarray, "n 5"] | None:
    """Convert range-Doppler and angle indices to point cloud data.

    Args:
        rd_indx: Range-Doppler indices.
        aoa_indx: Angle of Arrival indices.
        snr: Signal-to-noise ratio.
        p: AoAparam object containing radar parameters.

    Returns:
        Point cloud data as a numpy array [x, y, z, snr, velocity].
    """
    ang_azi = np.arcsin(p.vec_angle[aoa_indx[0]] / (2 * np.pi * p.d))
    ang_ele = np.arcsin(p.vec_angle[aoa_indx[1]] / (2 * np.pi * p.d))
    valid = (
        (ang_azi >= p.azimuth_fov[0])
        & (ang_azi <= p.azimuth_fov[1])
        & (ang_ele >= p.elevation_fov[0])
        & (ang_ele <= p.elevation_fov[1])
    )

    if not valid:
        return None

    depth = rd_indx[0] * p.range_bin_size
    velocity = (rd_indx[1] - p.doppler_fft_size / 2) * p.velocity_bin_size

    pcd = np.zeros(5, np.float32)
    pcd[0] = depth * np.cos(ang_azi * -1) * np.cos(ang_ele)
    pcd[1] = -depth * np.sin(ang_azi * -1) * np.cos(ang_ele)
    pcd[2] = -depth * np.sin(ang_ele * -1)
    pcd[3] = snr
    pcd[4] = velocity

    return pcd


@nb.njit
def detect_single(
    sig_azi: Complex[np.ndarray, "azi"],
    sig_2d: Complex[np.ndarray, "azi ele"],
    rd_indx: Int[np.ndarray, "2"],
    snr: Float[np.ndarray, ""],
    p: AoAparam,
) -> tuple[Bool[np.ndarray, "n_pc_preserve"], Float[np.ndarray, "n_pc_preserve 5"]]:
    """Detect point cloud for a single range-Doppler bin.

    Args:
        sig_azi: 1D azimuth signal.
        sig_2d: 2D angle signal.
        rd_indx: Range-Doppler indices.
        snr: Signal-to-noise ratio.
        p: AoAparam object containing radar parameters.
    """
    pc_count = 0
    pc_det = np.zeros((p.n_pc_preserve, 5), dtype=np.float32)
    mask = np.zeros((p.n_pc_preserve), dtype=np.bool)
    indx_azi = peak_detect(sig_azi, p.gamma, p.sidelobe_azi_db)
    if indx_azi.shape[0] != 0:
        for i_azi in indx_azi:
            indx_ele = peak_detect(sig_2d[i_azi[0], :], p.gamma, p.sidelobe_ele_db)
            if indx_ele.shape[0] != 0:
                for i_ele in indx_ele:
                    aoa_indx = [i_azi[0], i_ele[0]]
                    pc = to_xyzsv(rd_indx, aoa_indx, snr, p)
                    if pc is not None and pc_count < p.n_pc_preserve:
                        mask[pc_count] = True
                        pc_det[pc_count] = pc
                        pc_count += 1

    return mask, pc_det


class AOAPC:
    """Angle of Arrival Point Cloud Processing Class.

    Args:
        radar_param: RadarParam object containing radar parameters.
        gamma_db: Threshold factor in dB for peak detection.
        sidelobe_azi_db: Sidelobe threshold in dB for azimuth peak detection.
        sidelobe_ele_db: Sidelobe threshold in dB for elevation peak detection.
        azimuth_fov: Tuple defining the azimuth field of view in degrees.
        elevation_fov: Tuple defining the elevation field of view in degrees.
        aoa_fft_size: Size of the AoA FFT.
        n_pc_preserve: Number of point clouds to preserve per detection.
    """

    def __init__(
        self,
        radar_param: RadarParam,
        gamma_db: float = 0.2,
        sidelobe_azi_db: float = 1.0,
        sidelobe_ele_db: float = 0.0,
        azimuth_fov: Tuple[float, float] = (-80.0, 80.0),
        elevation_fov: Tuple[float, float] = (-20.0, 20.0),
        aoa_fft_size: int = 256,
        n_pc_preserve: int = 10,
    ):
        self.param = AoAparam(
            gamma=10 ** (gamma_db / 10),
            d=radar_param.antenna_dis,  # antenna distance in terms of wavelength
            range_bin_size=radar_param.rangeBinSize,
            velocity_bin_size=radar_param.velocityBinSize,
            doppler_fft_size=radar_param.DopplerFFTSize,
            sidelobe_azi_db=sidelobe_azi_db,
            sidelobe_ele_db=sidelobe_ele_db,
            n_pc_preserve=n_pc_preserve,
            azimuth_fov=np.deg2rad(azimuth_fov, dtype=np.float32),
            elevation_fov=np.deg2rad(elevation_fov, dtype=np.float32),
            vec_angle=np.linspace(
                -np.pi, np.pi, aoa_fft_size, endpoint=False, dtype=np.float32
            ),
        )

    @staticmethod
    @nb.njit(parallel=True)
    def aoa_pc(
        sig_azi: Complex[np.ndarray, "n azi"],
        sig_2d: Complex[np.ndarray, "n azi ele"],
        rd_indx: Int[np.ndarray, "n 2"],
        snr: Float[np.ndarray, "n"],
        p: AoAparam,
    ):
        n_detect = sig_azi.shape[0]

        pc_all = np.zeros((n_detect, p.n_pc_preserve, 5), dtype=np.float32)
        pc_mask = np.zeros((n_detect, p.n_pc_preserve), dtype=np.bool)

        for i in nb.prange(n_detect):
            mask, pc = detect_single(sig_azi[i], sig_2d[i], rd_indx[i], snr[i], p)
            pc_mask[i], pc_all[i] = mask, pc

        return pc_all, pc_mask

    def __call__(
        self,
        signal_azi: Complex[np.ndarray, "azi"],
        signal_angle: Complex[np.ndarray, "azi ele"],
        rd_indx: Int[np.ndarray, "2"],
        snr: Float[np.ndarray, "1"],
    ):
        pc_all, pc_mask = self.aoa_pc(
            signal_azi, signal_angle, rd_indx, snr, self.param
        )
        idx = np.argwhere(pc_mask)
        pc = pc_all[idx[:, 0], idx[:, 1]]
        return pc
