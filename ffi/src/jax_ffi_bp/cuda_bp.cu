/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

__device__ std::complex<float> complex_multiply(std::complex<float> a, std::complex<float> b)
{
  float cr = a.real() * b.real() - a.imag() * b.imag();
  float ci = a.real() * b.imag() + a.imag() * b.real();
  return std::complex<float>(cr, ci);
}

__global__ void BPKernel(std::complex<float> *sig, float *range, float *angle, float *k,
                         float *min_dis, float *max_dis, float *azi_fov,
                         std::complex<float> *out, size_t rows, size_t cols)
{
  size_t i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
  size_t j = blockIdx.x * blockDim.x + threadIdx.x; // Column index

  if (i < rows && j < cols)
  {
    if (angle[i] < 0 || angle[i] > azi_fov[i])
    {
      out[i * j + j] = std::complex<float>(0, 0);
      return;
    }

    if (range[i] < min_dis[i] || range[i] > max_dis[i])
    {
      out[i * j + j] = std::complex<float>(0, 0);
      return;
    }

    std::complex<float> c(cos(-2 * k[i] * range[i]), sin(-2 * k[i] * range[i]));
    out[i * j + j] = complex_multiply(sig[i * j + j], c);
  }
}

ffi::Error BPHost(cudaStream_t stream, ffi::Buffer<ffi::C64> sig,
                  ffi::Buffer<ffi::F32> range, ffi::Buffer<ffi::F32> angle, ffi::Buffer<ffi::F32> k,
                  ffi::Buffer<ffi::F32> min_dis, ffi::Buffer<ffi::F32> max_dis, ffi::Buffer<ffi::F32> azi_fov,
                  ffi::ResultBuffer<ffi::C64> out)
{
  auto dim = min_dis.dimensions();
  size_t cols = dim[1];
  size_t rows = dim[0];
  dim3 blockDim(32, 4); // 32 threads for columns, 4 threads for rows
  dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

  BPKernel<<<gridDim, blockDim, /*shared_mem=*/0, stream>>>(
      sig.typed_data(), range.typed_data(), angle.typed_data(), k.typed_data(),
      min_dis.typed_data(), max_dis.typed_data(), azi_fov.typed_data(),
      out->typed_data(), rows, cols);

  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error(
        XLA_FFI_Error_Code_INTERNAL,
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BackProject, BPHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
        .Arg<ffi::Buffer<ffi::C64>>()             // sig
        .Arg<ffi::Buffer<ffi::F32>>()             // range
        .Arg<ffi::Buffer<ffi::F32>>()             // angle
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::C64>>(),            // out
    {xla::ffi::Traits::kCmdBufferCompatible}); // cudaGraph enabled
