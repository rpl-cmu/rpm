#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cufft.h>
#include <thrust/complex.h>
#include "xla/ffi/api/ffi.h"
#include "aoa.hpp"

namespace ffi = xla::ffi;
namespace cg = cooperative_groups;

using tc_type = typename thrust::complex<float>;

#define AFFT_SIZE 256
#define BLOCK_DIM 256

__global__ void calibrate_rearange_kernel(
    const std::complex<float> *sig, // [N, rx, tx]
    const int *rd_idx,              // [N, 2]
    const size_t N,                 // cfar detect count
    const size_t n_rx,              // rx count
    const size_t n_tx,              // tx count
    const size_t dfftsize,          // Doppler FFT size
    const int *layout,              // [rx, tx, 2]
    const int *mask,                // [rx, tx] antenna mask
    const size_t afft_size,         // fft_size
    std::complex<float> *out        // [N, fft_size, fft_size]
)
{
    size_t idx = cg::this_grid().thread_rank();

    // parallel over all sig
    if (idx >= N * n_rx * n_tx)
        return;

    int i_n = idx / (n_rx * n_tx);
    int i_rx = (idx % (n_rx * n_tx)) / n_tx;
    int i_tx = idx % n_tx;

    if (!mask[i_rx * n_tx + i_tx])
        return;

    tc_type s(sig[idx]);

    // Doppler phase correction due to TDM MIMO
    size_t d_idx = rd_idx[2 * i_n + 1];
    float deltaPhi = 2 * M_PI * (d_idx - dfftsize / 2) / (n_tx * dfftsize);
    s *= thrust::exp(tc_type(0, -1 * i_tx * deltaPhi));

    // rearrange phasor
    // save as [N, le, la] on [N, afft_size, afft_size]
    size_t i_a = layout[i_rx * (n_tx * 2) + i_tx * 2 + 0];
    size_t i_e = layout[i_rx * (n_tx * 2) + i_tx * 2 + 1];
    size_t o_idx = i_n * afft_size * afft_size + i_e * afft_size + i_a;
    out[o_idx] = std::complex<float>(s.real(), s.imag());
}

__global__ void shift_transpose(
    const std::complex<float> *A,   // [N, fftsize(ele), fftsize(azi)]
    const size_t N,                 // cfar detect count
    const size_t afft_size,         // fft size
    std::complex<float> *A_shift_T, // [N, fftsize(azi), fftsize(ele)]
    std::complex<float> *azi_fft    // [N, fftsize(azi)]
)
{
    size_t idx = cg::this_grid().thread_rank();

    // parallel process all dim
    if (idx >= N * afft_size * afft_size)
        return;

    int i_n = idx / (afft_size * afft_size);
    int i_e = (idx % (afft_size * afft_size)) / afft_size;
    int i_a = idx % afft_size;

    // fft shift i_a
    int o_a = (i_a + afft_size / 2) % afft_size;
    int o_e = i_e;

    size_t o_idx = i_n * afft_size * afft_size + o_a * afft_size + o_e;

    A_shift_T[o_idx] = A[idx];

    // copy azimuth at 0 elevation
    if (o_e == 0)
    {
        azi_fft[i_n * afft_size + o_a] = A[idx];
    }
}

__global__ void AoA_kernel(
    std::complex<float> *sig_azi, // [N, afft_szie(azi)]
    std::complex<float> *sig_2D,  // [N, afft_szie(azi), afft_szie(ele)]
    const int *rd_idx,            // [N, 2]
    const float *snrs,            // [N]
    const size_t N,               // cfar detect count
    const size_t dfftsize,        //
    const float gamma,            // peak diff dB
    const float sidelobe_azi_dB,  //
    const float sidelobe_ele_dB,  //
    const float azimuth_min,      //
    const float azimuth_max,      //
    const float elevation_min,    //
    const float elevation_max,    //
    const int n_pc_preserve,      //
    const float r_res,            //
    const float d_res,            //
    const float antenna_dis,      //
    bool *pc_mask,                // [N, n_pc_preserve]
    float *pc_all                 // [N, n_pc_preserve, 5]
)
{
    size_t idx = cg::this_grid().thread_rank();

    // parallel over N dimension
    if (idx >= N)
        return;

    bool mask_azi[AFFT_SIZE], mask_ele[AFFT_SIZE];
    std::complex<float> ele[AFFT_SIZE];
    int pc_count = 0;

    peak_detect(sig_azi, idx * AFFT_SIZE, AFFT_SIZE,
                gamma, sidelobe_azi_dB, mask_azi);

    for (size_t a = 0; a < AFFT_SIZE; a++)
    {
        if (mask_azi[a])
        {
            size_t i_e = idx * AFFT_SIZE * AFFT_SIZE + a * AFFT_SIZE;
            fftshift(sig_2D, i_e, AFFT_SIZE);
            peak_detect(sig_2D, i_e, AFFT_SIZE, gamma, sidelobe_ele_dB, mask_ele);

            for (size_t e = 0; e < AFFT_SIZE; e++)
            {
                if (mask_ele[e])
                {
                    to_xyzsv(rd_idx[idx * 2 + 0], rd_idx[idx * 2 + 1], a, e, snrs[idx],
                             antenna_dis, azimuth_min, azimuth_max, elevation_min, elevation_max,
                             AFFT_SIZE, dfftsize, r_res, d_res, n_pc_preserve, idx, pc_count,
                             pc_mask, pc_all);

                    if (pc_count >= n_pc_preserve)
                        return;
                }
                mask_ele[e] = false;
            }
        }
    }
}

ffi::Error AoAHost(cudaStream_t stream,
                   ffi::Buffer<ffi::C64> sig,            // [N, rx, tx]
                   ffi::Buffer<ffi::S32> rd_idx,         // [N, 2]
                   ffi::Buffer<ffi::F32> snrs,           // [N]
                   ffi::Buffer<ffi::S32> layout,         // [rx, tx, 2]
                   ffi::Buffer<ffi::S32> mask,           // [rx, tx]
                   ffi::ResultBuffer<ffi::PRED> pc_mask, // [N, n_pre] bool
                   ffi::ResultBuffer<ffi::F32> pc_all,   // [N, n_pre, 5]
                   const size_t dfft_size,               //
                   const float range_res,                //
                   const float doppler_res,              //
                   const float antenna_dis,              //
                   const float gamma_dB,                 // peak diff dB
                   const float sidelobe_azi_dB,          //
                   const float sidelobe_ele_dB,          //
                   const float azimuth_min,              //
                   const float azimuth_max,              //
                   const float elevation_min,            //
                   const float elevation_max,            //
                   const size_t n_pc_preserve            //
)
{
    ffi::AnyBuffer::Dimensions dim = sig.dimensions();
    size_t N = dim[0];
    size_t n_rx = dim[1];
    size_t n_tx = dim[2];
    int block_dim, grid_dim;
    block_dim = BLOCK_DIM;

    std::complex<float> *reshaped, *fft_cache, *azi_cache;

    size_t size2d = N * AFFT_SIZE * AFFT_SIZE * sizeof(std::complex<float>);
    size_t size1d = N * AFFT_SIZE * sizeof(std::complex<float>);

    cudaMalloc((void **)&fft_cache, size2d);
    cudaMalloc((void **)&azi_cache, size1d);
    cudaMalloc((void **)&reshaped, size2d);

    cudaMemset(reshaped, 0, size2d);
    cudaMemset(fft_cache, 0, size2d);
    cudaMemset(pc_mask->typed_data(), 0, pc_mask->size_bytes());
    cudaMemset(pc_all->typed_data(), 0, pc_all->size_bytes());

    cufftHandle plan, plan2;
    cufftPlan1d(&plan, AFFT_SIZE, CUFFT_C2C, N * AFFT_SIZE);
    cufftPlan1d(&plan2, AFFT_SIZE, CUFFT_C2C, N * AFFT_SIZE);

    // parallel process all sig dim
    grid_dim = (N * n_rx * n_tx + block_dim - 1) / block_dim;
    calibrate_rearange_kernel<<<grid_dim, block_dim, 0, stream>>>(
        sig.typed_data(), rd_idx.typed_data(), N, n_rx, n_tx,
        dfft_size, layout.typed_data(), mask.typed_data(),
        AFFT_SIZE, reshaped);

    // 1D fft on azi dim
    cufftSetStream(plan, stream);
    cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(reshaped),
                 reinterpret_cast<cufftComplex *>(fft_cache), CUFFT_FORWARD);

    // parallel process all fft dim
    grid_dim = (N * AFFT_SIZE * AFFT_SIZE + block_dim - 1) / block_dim;
    shift_transpose<<<grid_dim, block_dim, 0, stream>>>(
        fft_cache, N, AFFT_SIZE, reshaped, azi_cache);

    // 1D fft on ele dim
    cufftSetStream(plan2, stream);
    cufftExecC2C(plan2, reinterpret_cast<cufftComplex *>(reshaped),
                 reinterpret_cast<cufftComplex *>(fft_cache), CUFFT_FORWARD);

    // parallel over N dim
    grid_dim = (N + block_dim - 1) / block_dim;
    AoA_kernel<<<grid_dim, block_dim, 0, stream>>>(
        azi_cache, fft_cache, rd_idx.typed_data(), snrs.typed_data(),
        N, dfft_size, powf(10.0f, gamma_dB / 10.0f), sidelobe_azi_dB, sidelobe_ele_dB,
        azimuth_min, azimuth_max, elevation_min, elevation_max, n_pc_preserve, range_res, doppler_res,
        antenna_dis, pc_mask->typed_data(), pc_all->typed_data());

    cudaStreamSynchronize(stream);
    cufftDestroy(plan);
    cufftDestroy(plan2);
    cudaFree(reshaped);
    cudaFree(fft_cache);
    cudaFree(azi_cache);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess)
    {
        return ffi::Error::Internal(
            std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

// Creates symbol with C linkage that can be loaded using Python ctypes
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AoA, AoAHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
        .Arg<ffi::Buffer<ffi::C64>>()             // sig
        .Arg<ffi::Buffer<ffi::S32>>()             // rd_idx
        .Arg<ffi::Buffer<ffi::F32>>()             // snrs
        .Arg<ffi::Buffer<ffi::S32>>()             // layout
        .Arg<ffi::Buffer<ffi::S32>>()             // mask
        .Ret<ffi::Buffer<ffi::PRED>>()            // pc_mask
        .Ret<ffi::Buffer<ffi::F32>>()             // pc_all
        .Attr<size_t>("dfft_size")                // dfft size
        .Attr<float>("range_res")                 //
        .Attr<float>("doppler_res")               //
        .Attr<float>("antenna_dis")               //
        .Attr<float>("gamma_dB")                  // peak diff dB
        .Attr<float>("sidelobe_azi_dB")           //
        .Attr<float>("sidelobe_ele_dB")           //
        .Attr<float>("azimuth_min")               //
        .Attr<float>("azimuth_max")               //
        .Attr<float>("elevation_min")             //
        .Attr<float>("elevation_max")             //
        .Attr<size_t>("n_pc_preserve"),           //
    {xla::ffi::Traits::kCmdBufferCompatible});    // cudaGraph enabled
