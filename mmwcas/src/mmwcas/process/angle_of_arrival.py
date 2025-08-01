import numpy as np
import numba as nb

from typing import Tuple
from jaxtyping import Complex, Array, Float, Int
from ..dataset import RadarParam
from .aoa_param import AoAparam

nb.config.THREADING_LAYER = "omp" # type: ignore


@nb.njit
def peak_detect(
    signal: Complex[Array, "n"],
    gamma: float,
    sidelobe_dB: float,
):
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

    sideLobeThresh = np.max(spectrum) * (10 ** (-sidelobe_dB / 10))
    pkea_mask = np.logical_and(spectrum >= sideLobeThresh, max_locs)
    return np.argwhere(pkea_mask)


@nb.njit
def to_xyzsv(
    rd_indx: Int[Array, "2"] | list,
    aoa_indx: Int[Array, "2"] | list,
    snr: Float[Array, "1"],
    p: AoAparam,
):
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

    depth = rd_indx[0] * p.rangeBinSize
    velocity = (rd_indx[1] - p.dopplerFFTSize / 2) * p.velocityBinSize

    pcd = np.zeros(5, np.float32)
    pcd[0] = depth * np.cos(ang_azi * -1) * np.cos(ang_ele)
    pcd[1] = -depth * np.sin(ang_azi * -1) * np.cos(ang_ele)
    pcd[2] = -depth * np.sin(ang_ele * -1)
    pcd[3] = snr
    pcd[4] = velocity

    return pcd


@nb.njit
def detect_single(
    sig_azi: Complex[Array, "azi"],
    sig_2D: Complex[Array, "azi ele"],
    rd_indx: Int[Array, "2"],
    snr: Float[Array, ""],
    p: AoAparam,
):
    pc_count = 0
    pc_det = np.zeros((p.n_pc_preserve, 5), dtype=np.float32)
    mask = np.zeros((p.n_pc_preserve), dtype=np.bool)
    indx_azi = peak_detect(sig_azi, p.gamma, p.sidelobe_azi_dB)
    if indx_azi.shape[0] != 0:
        for i_azi in indx_azi:
            indx_ele = peak_detect(sig_2D[i_azi[0], :], p.gamma, p.sidelobe_ele_dB)
            if indx_ele.shape[0] != 0:
                for i_ele in indx_ele:
                    aoa_indx = [i_azi[0], i_ele[0]]
                    pc = to_xyzsv(rd_indx, aoa_indx, snr, p)
                    if pc is not None and pc_count < p.n_pc_preserve:
                        mask[pc_count] = True
                        pc_det[pc_count] = pc
                        pc_count += 1

    return mask, pc_det


class AOA_PC:
    def __init__(
        self,
        radar_param: RadarParam,
        gamma_dB: float = 0.2,
        sidelobe_azi_dB: float = 1.0,
        sidelobe_ele_dB: float = 0.0,
        azimuth_fov: Tuple[float, float] = (-80.0, 80.0),
        elevation_fov: Tuple[float, float] = (-20.0, 20.0),
        aoa_fft_size: int = 256,
        n_pc_preserve: int = 10,
    ):
        self.param = AoAparam(
            gamma=10 ** (gamma_dB / 10),
            d=radar_param.antenna_dis,  # antenna distance in terms of wavelength
            rangeBinSize=radar_param.rangeBinSize,
            velocityBinSize=radar_param.velocityBinSize,
            dopplerFFTSize=radar_param.DopplerFFTSize,
            sidelobe_azi_dB=sidelobe_azi_dB,
            sidelobe_ele_dB=sidelobe_ele_dB,
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
        sig_azi: Complex[Array, "n azi"],
        sig_2D: Complex[Array, "n azi ele"],
        rd_indx: Int[Array, "n 2"],
        snr: Float[Array, "n"],
        p: AoAparam,
    ):
        n_detect = sig_azi.shape[0]

        pc_all = np.zeros((n_detect, p.n_pc_preserve, 5), dtype=np.float32)
        pc_mask = np.zeros((n_detect, p.n_pc_preserve), dtype=np.bool)

        for i in nb.prange(n_detect):
            mask, pc = detect_single(sig_azi[i], sig_2D[i], rd_indx[i], snr[i], p)
            pc_mask[i], pc_all[i] = mask, pc

        return pc_all, pc_mask

    def __call__(
        self,
        signal_azi: Complex[Array, "azi"],
        signal_angle: Complex[Array, "azi ele"],
        rd_indx: Int[Array, "2"],
        snr: Float[Array, "1"],
    ):
        pc_all, pc_mask = self.aoa_pc(
            signal_azi, signal_angle, rd_indx, snr, self.param
        )
        idx = np.argwhere(pc_mask)
        pc = pc_all[idx[:, 0], idx[:, 1]]
        return pc
