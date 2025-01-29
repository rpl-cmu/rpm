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

from absl.testing import absltest

import numpy as np
import jax
import jax.numpy as jnp

from jax_ffi_bp import cuda_bp

bp_cuda = cuda_bp.BackProjectCUDA

# dim = 10000
# sig = jnp.ones((dim, dim)) + 1j * jnp.ones((dim, dim))
# dis = jnp.ones_like(sig, dtype=jnp.float32)
# ang = jnp.ones_like(sig, dtype=jnp.float32)

sig = jnp.array([1, 2, 3, 4])
sig = sig + 1j * sig
dis = jnp.array([0.1, 3.0, 4.0, 5.0])
ang = jnp.array([1.0, 2.0, 3.0, 50.0])

min_dis = jnp.array([0.6])
max_dis = jnp.array([10.0])
azi_fov = jnp.array([40.0])
k = jnp.array([10.0])

bp = jax.jit(bp_cuda)

out = bp(sig, dis, ang, k, min_dis, max_dis, azi_fov)
print(out.shape)
print(out)


mask_d = jnp.logical_and(dis > min_dis, dis < max_dis)
mask_a = jnp.logical_and(ang > 0, ang < azi_fov)
mask = jnp.logical_and(mask_a, mask_d)
out = sig * jnp.exp(-1j * 2 * k * dis) * mask
print(out)
