#include <cuda_runtime.h>
#include <complex>
#include <thrust/complex.h>

__device__ void peak_detect(
    const std::complex<float> *sig, // continous on least significant dimension
    size_t idx,                     //
    const size_t sig_len,           //
    const float gamma,              //
    const float sidelobe_dB,        //
    bool *mask                      //
)
{
    float minval = FLT_MAX, maxval = 0, absmax = 0;
    int maxloc = 0, extendloc = 0;
    bool locatemax = false, initstage = true;

    int i = 0;
    while (i < (sig_len + extendloc - 1))
    {
        int iloc = i % sig_len;
        float val = thrust::abs(thrust::complex<float>(sig[idx + iloc]));

        if (val > absmax)
            absmax = val;

        if (val > maxval)
        {
            maxval = val;
            maxloc = iloc;
        }

        if (val < minval)
            minval = val;

        if (locatemax)
        {
            if (val < (maxval / gamma))
            {
                mask[maxloc] = true;
                minval = val;
                locatemax = false;
            }
        }
        else
        {
            if (val > (minval * gamma))
            {
                locatemax = true;
                maxval = val;
                if (initstage)
                {
                    extendloc = i;
                    initstage = false;
                }
            }
        }

        i++;
    }

    float thresh = absmax * powf(10.0f, (-sidelobe_dB / 10.0f));
    for (size_t k = 0; k < sig_len; k++)
    {
        if (mask[k])
        {
            if (thrust::abs(thrust::complex<float>(sig[idx + k])) < thresh)
                mask[k] = false;
        }
    }
}

__device__ void fftshift(std::complex<float> *sig2D, size_t idx, size_t fftsize)
{
    std::complex<float> tmp;
    for (size_t i = 0; i < fftsize / 2; i++)
    {
        tmp = sig2D[idx + i];
        sig2D[idx + i] = sig2D[idx + (i + fftsize / 2)];
        sig2D[idx + (i + fftsize / 2)] = tmp;
    }
}

__device__ void to_xyzsv(
    const int r,
    const int d,
    const int a,
    const int e,
    const float snr,
    const float antenna_dis,
    const float azi_min,
    const float azi_max,
    const float ele_min,
    const float ele_max,
    const int afft_size,
    const int dfft_size,
    const float r_res,
    const float d_res,
    const int n_pc_preserve,
    const size_t n, // current N
    int &pc_count,  // pc_count < n_pc_preserve
    bool *pc_mask,  // [N, n_pc_preserve]
    float *pc_all   // [N, n_pc_preserve, 5]
)
{
    float ang_azi = asin((-M_PI + a * 2 * M_PI / afft_size) / (2 * M_PI * antenna_dis));
    float ang_ele = asin((-M_PI + e * 2 * M_PI / afft_size) / (2 * M_PI * antenna_dis));

    if (ang_azi < azi_min || ang_azi > azi_max || ang_ele < ele_min || ang_ele > ele_max)
        return;

    float range = r * r_res;
    float velocity = (d - dfft_size / 2) * d_res;

    size_t i = n * n_pc_preserve * 5 + pc_count * 5;
    pc_all[i + 0] = range * cos(-ang_azi) * cos(ang_ele);
    pc_all[i + 1] = -range * sin(-ang_azi) * cos(ang_ele);
    pc_all[i + 2] = -range * sin(-ang_ele);
    pc_all[i + 3] = snr;
    pc_all[i + 4] = velocity;

    pc_mask[n * n_pc_preserve + pc_count] = true;
    pc_count++;
}