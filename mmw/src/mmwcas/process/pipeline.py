import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Callable, Optional
from functools import cached_property
from jaxtyping import Array, Complex, Float, Bool
from jax.scipy.signal import correlate2d

from .range_fft import RangeFFT
from .doppler_fft import DopplerFFT
from .cfar import CFAR, CFAR_CASO
from .azimuth_fft import AzimuthFFT
from .angle_fft import AngleFFT
from .angle_of_arrival import AOA_PC
from ..dataset.params import RadarParam
from ..ffi.aoa import AoA_CU


class RangeProc:
    def __init__(self, radar_param: RadarParam, normalize_factor=5e4):
        self.range_proc = RangeFFT(
            fft_size=radar_param.rangeFFTSize, adc_samples=radar_param.numADCSample
        )
        self.normalize_factor = normalize_factor

    def __call__(
        self, sig_tensor: Complex[Array, "sample chirp rx tx"]
    ) -> Float[Array, "range chirp"]:
        sig_tensor = self.range_proc(sig_tensor)
        r, d, rx, tx = sig_tensor.shape
        sig_tensor = sig_tensor.reshape(r, d, -1)
        range_img = jnp.mean(jnp.abs(sig_tensor), axis=-1)

        range_img = jnp.clip(range_img / self.normalize_factor, 0, 1)

        return range_img


class RangeAzimuthProc:
    def __init__(
        self,
        radar_param: RadarParam,
        angele_fft_size=256,
        range_bin_min=5,
        range_bin_max=236,
        angle_3dB=85.0,
        normalize_factor=10,
        output_normalize=True,
    ):
        self.range_proc = RangeFFT(
            fft_size=radar_param.rangeFFTSize,
            adc_samples=radar_param.numADCSample,
            norm="forward",
        )

        angle_bin_min = int((90 - angle_3dB) / 90 * (angele_fft_size // 2))
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
        sig_tensor = self.range_proc(sig_tensor)
        # sig_tensor = self.doppler_proc(sig_tensor)
        sig_tensor = self.azimuth_proc(sig_tensor)

        range_azimuth = jnp.mean(jnp.abs(sig_tensor), axis=1)

        if self.output_normalize:
            return jnp.clip(range_azimuth / self.normalize_factor, 0, 1)

        return range_azimuth


class RangeAzimuthCFARProc:
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
            rangeFFTSize=radar_param.rangeFFTSize,
            DopplerFFTSize=radar_param.DopplerFFTSize,
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
    ) -> tuple[Bool[Array, "range doppler"] | None, Float[Array, "range doppler"]]:
        sig_tensor = self.range_proc(sig_tensor)
        sig_tensor = self.doppler_proc(sig_tensor)

        # obj_mask, rd_image, snr = self.cfar(sig_tensor)
        # rd_image = jnp.clip(rd_image / self.normalize_factor, 0, 1)

        s_r, s_d, _, _ = sig_tensor.shape
        range_dopp = sig_tensor.reshape(s_r, s_d, -1)

        # non-coherent signal combination along the antenna array
        rd_image = jnp.mean(jnp.abs(range_dopp), axis=-1)
        rd_image = jnp.clip(rd_image / self.normalize_factor, 0, 1)

        return None, rd_image


class RadarCubeProc:
    def __init__(
        self, radar_param: RadarParam, angle_fft_size: Optional[Tuple[int, int]] = None
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
                np.max(radar_param.antenna_azi) + 1,
                np.max(radar_param.antenna_ele) + 1,
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

    def __call__(self, sig_tensor: Complex[Array, "sample chirp rx tx"]):
        sig_tensor = self.range_proc(sig_tensor)
        sig_tensor = self.doppler_proc(sig_tensor)
        sig_tensor = self.afft_proc(sig_tensor, self.rd_all)
        return sig_tensor


class RadarPCProc:
    def __init__(
        self,
        radar_param: RadarParam,
        train_range: int = 8,
        guard_range: int = 8,
        train_doppler: int = 4,
        guard_doppler: int = 0,
        K0_range: float = 5.0,
        K0_doppler: float = 3.0,
        discard_range_close: int = 10,
        discard_range_far: int = 20,
        angle_fft_size: int = 256,
        azimuth_fov: Tuple[float, float] = (-80.0, 80.0),
        elevation_fov: Tuple[float, float] = (-20.0, 20.0),
        aoa_gamma_dB: float = 0.2,
        sidelobe_azi_dB: float = 1.0,
        sidelobe_ele_dB: float = 0.0,
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
        self.cfar = CFAR_CASO(
            train_range=train_range,
            guard_range=guard_range,
            train_doppler=train_doppler,
            guard_doppler=guard_doppler,
            K0_range=K0_range,
            K0_doppler=K0_doppler,
            discard_range_close=discard_range_close,
            discard_range_far=discard_range_far,
        )
        self.angle_fft_size = angle_fft_size
        afft = AngleFFT(
            radar_param=radar_param, fft_size=(angle_fft_size, angle_fft_size)
        )
        self.angle_fft = jax.jit(jax.vmap(afft))

        self.aoa_pc = AOA_PC(
            radar_param=radar_param,
            gamma_dB=aoa_gamma_dB,
            sidelobe_azi_dB=sidelobe_azi_dB,
            sidelobe_ele_dB=sidelobe_ele_dB,
            azimuth_fov=azimuth_fov,
            elevation_fov=elevation_fov,
            aoa_fft_size=angle_fft_size,
        )

    @cached_property
    def fft_procs(self):
        @jax.jit
        def __inner__(sig_tensor: Complex[Array, "sample chirp rx tx"]):
            rfft = self.range_proc(sig_tensor)
            rdfft = self.doppler_proc(rfft)
            obj_mask, _, snr = self.cfar(rdfft)
            return obj_mask, rdfft, snr

        return __inner__

    def __call__(self, sig_tensor: Complex[Array, "sample chirp rx tx"]):

        obj_mask, rdfft, snrs = self.fft_procs(sig_tensor)
        sig_pick = np.asarray(rdfft)[obj_mask]
        objs = np.argwhere(np.asarray(obj_mask))
        snrs = np.asarray(snrs)[obj_mask]

        sig_azis, sig_angles = self.angle_fft(sig_pick, objs)
        pc = self.aoa_pc(np.asarray(sig_azis), np.asarray(sig_angles), objs, snrs)

        return np.asarray(pc)


class RadarPCProcCU:
    def __init__(
        self,
        radar_param: RadarParam,
        train_range=8,
        guard_range=8,
        train_doppler=4,
        guard_doppler=0,
        K0_range=5.0,
        K0_doppler=3.0,
        discard_range_close=10,
        discard_range_far=20,
        azimuth_fov=(-80.0, 80.0),
        elevation_fov=(-20.0, 20.0),
        aoa_gamma_dB=0.2,
        sidelobe_azi_dB=1.0,
        sidelobe_ele_dB=0.0,
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
        self.cfar = CFAR_CASO(
            train_range=train_range,
            guard_range=guard_range,
            train_doppler=train_doppler,
            guard_doppler=guard_doppler,
            K0_range=K0_range,
            K0_doppler=K0_doppler,
            discard_range_close=discard_range_close,
            discard_range_far=discard_range_far,
        )

        self.layout = jnp.stack(
            (radar_param.antenna_azi, radar_param.antenna_ele), axis=-1
        )
        self.nla = np.max(radar_param.antenna_azi) + 1
        self.nle = np.max(radar_param.antenna_ele) + 1

        ant_map = -np.ones((self.nla, self.nle), dtype=np.int32)
        idx = self.layout.reshape(-1, 2)
        ant_map[idx[:, 0], idx[:, 1]] = np.arange(radar_param.numRx * radar_param.numTx)
        self.ant_mask = np.zeros(radar_param.numRx * radar_param.numTx, dtype=np.int32)
        self.ant_mask[ant_map[ant_map >= 0]] = 1
        self.ant_mask = jnp.asarray(self.ant_mask)

        self.doppler_fft_size = radar_param.DopplerFFTSize
        self.range_res = radar_param.rangeBinSize
        self.doppler_res = radar_param.velocityBinSize
        self.antenna_dis = radar_param.antenna_dis
        self.gamma_dB = aoa_gamma_dB
        self.sidelobe_azi_dB = sidelobe_azi_dB
        self.sidelobe_ele_dB = sidelobe_ele_dB
        self.azimuth_fov = azimuth_fov
        self.elevation_fov = elevation_fov
        self.n_pc_preserve = 10

    @cached_property
    def fft_procs(self):
        @jax.jit
        def __inner__(sig_tensor: Complex[Array, "sample chirp rx tx"]):
            rfft = self.range_proc(sig_tensor)
            rdfft = self.doppler_proc(rfft)
            obj_mask, _, snr = self.cfar(rdfft)
            return obj_mask, rdfft, snr

        return __inner__

    def __call__(self, sig_tensor: Complex[Array, "sample chirp rx tx"]):

        obj_mask, rdfft, snrs = self.fft_procs(sig_tensor)

        sig_tensor = np.asarray(rdfft)[obj_mask]
        objs = np.argwhere(np.asarray(obj_mask)).astype(np.int32)
        snrs = np.asarray(snrs)[obj_mask]

        pc_mask, pc_all = AoA_CU(
            sig_tensor,
            objs,
            snrs,
            self.layout,
            self.ant_mask,
            self.doppler_fft_size,
            self.range_res,
            self.doppler_res,
            self.antenna_dis,
            self.gamma_dB,
            self.sidelobe_azi_dB,
            self.sidelobe_ele_dB,
            self.azimuth_fov,
            self.elevation_fov,
            self.n_pc_preserve,
        )

        mask = np.asarray(pc_mask)
        pc = np.asarray(pc_all)[mask]

        return pc
