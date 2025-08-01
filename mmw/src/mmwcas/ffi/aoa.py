import os
import ctypes

import numpy as np

import jax
import jax.numpy as jnp

from jaxtyping import Array, Complex, Int, Bool, Float

# Load the shared library with the FFI target definitions
SHARED_LIBRARY = os.path.join(os.path.dirname(__file__), "lib_aoa.so")
library = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)

jax.ffi.register_ffi_target("AoA", jax.ffi.pycapsule(library.AoA), platform="CUDA")

AFFT_SIZE = 256


def AoA_CU(
    sig: Complex[Array, "n rx tx"],
    rd_idx: Int[Array, "n 2"],
    snrs: Float[Array, "n"],
    layout: Int[Array, "rx tx 2"],
    mask: Int[Array, "rx tx"],
    dfft_size: int,
    range_res: float,
    doppler_res: float,
    antenna_dis: float,
    gamma_dB: float,
    sidelobe_azi_dB: float,
    sidelobe_ele_dB: float,
    azimuth_fov: tuple[float],  # degree
    elevation_fov: tuple[float],  # degree
    n_pc_pre: int,
):
    assert sig.dtype == jnp.complex64
    assert rd_idx.dtype == jnp.int32
    assert snrs.dtype == jnp.float32
    assert layout.dtype == jnp.int32
    assert mask.dtype == jnp.int32

    n = sig.shape[0]
    mask_type = jax.ShapeDtypeStruct((n, n_pc_pre), jnp.bool)
    pc_type = jax.ShapeDtypeStruct((n, n_pc_pre, 5), jnp.float32)

    pc_mask, pc_all = jax.ffi.ffi_call("AoA", (mask_type, pc_type))(
        sig,
        rd_idx,
        snrs,
        layout,
        mask,
        dfft_size=np.uint64(dfft_size),
        range_res=np.float32(range_res),
        doppler_res=np.float32(doppler_res),
        antenna_dis=np.float32(antenna_dis),
        gamma_dB=np.float32(gamma_dB),
        sidelobe_azi_dB=np.float32(sidelobe_azi_dB),
        sidelobe_ele_dB=np.float32(sidelobe_ele_dB),
        azimuth_min=np.deg2rad(np.float32(azimuth_fov[0])),
        azimuth_max=np.deg2rad(np.float32(azimuth_fov[1])),
        elevation_min=np.deg2rad(np.float32(elevation_fov[0])),
        elevation_max=np.deg2rad(np.float32(elevation_fov[1])),
        n_pc_preserve=np.uint64(n_pc_pre),
    )
    return pc_mask, pc_all
