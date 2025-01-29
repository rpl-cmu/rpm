# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An end-to-end example demonstrating the use of the JAX FFI with CUDA.

The specifics of the kernels are not very important, but the general structure,
and packaging of the extension are useful for testing.
"""

import os
import ctypes

import numpy as np

import jax
import jax.numpy as jnp

# Load the shared library with the FFI target definitions
SHARED_LIBRARY = os.path.join(os.path.dirname(__file__), "lib_cuda_bp.so")
library = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)

jax.ffi.register_ffi_target(
    "BackProject", jax.ffi.pycapsule(library.BackProject), platform="CUDA"
)


def BackProjectCUDA(sig, range, angle, k, min_dis, max_dis, azi_fov):
    assert sig.dtype == jnp.complex64
    assert range.dtype == jnp.float32
    assert angle.dtype == jnp.float32
    assert sig.shape == range.shape == angle.shape
    out_type = jax.ShapeDtypeStruct(sig.shape, sig.dtype)
    out = jax.ffi.ffi_call("BackProject", out_type, vmap_method="broadcast_all")(
        sig, range, angle, k, min_dis, max_dis, azi_fov
    )
    return out
