#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


/// (x + bias) * mask * scale + add


template <typename T>
struct MaskScaleAdd
{
    MaskScaleAdd(const uint8_t * mask,
                 float scale,
                 const T * add) :
        mask(mask),
        scale(scale),
        add(add)
    {

    }

    __device__ T operator()(T x, int i) const
    {
        return x * static_cast<T>(static_cast<bool>(mask[i]) * scale) + add[i];
    }

    const uint8_t * mask = nullptr;
    float scale = 0.0f;
    const T * add = nullptr;
};


template <typename Func, typename T>
__global__ void fusedBiasedMaskScaleAddNaive(const T * __restrict__ x,
                                             int nx,
                                             const T * __restrict__ bias,
                                             int biasSize,
                                             T * __restrict__ y,
                                             Func func)
{
    const int stride = gridDim.x * blockDim.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = gid ; i < nx; i += stride)
    {
        T tmp = x[i] + bias[i % biasSize];
        y[i] = func(tmp, i);
    }
}


template <typename Func, typename T>
__global__ void fusedBiasedMaskScaleAddSmem(const T * __restrict__ x,
                                            int nx,
                                            const T * __restrict__ glbBias,
                                            int biasSize,
                                            T * __restrict__ y,
                                            Func func)
{
    extern __shared__ T bias[];  // (biasSize,)

    const int stride = gridDim.x * blockDim.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // bias is a small size and loaded for multiple times.
    // Move from gmem to smem.
    if (tid < biasSize)
    {
        bias[tid] = glbBias[tid];
    }

    __syncthreads();

    for (int i = gid ; i < (nx >> 2); i += stride)
    {
        const float4 a = reinterpret_cast<const float4 * __restrict__>(x)[i];

        float4 b;
        b.x = func(a.x + bias[ (i << 2)      % biasSize],  i << 2);
        b.y = func(a.y + bias[((i << 2) + 1) % biasSize], (i << 2) + 1);
        b.z = func(a.z + bias[((i << 2) + 2) % biasSize], (i << 2) + 2);
        b.w = func(a.w + bias[((i << 2) + 3) % biasSize], (i << 2) + 3);

        reinterpret_cast<float4 * __restrict__>(y)[i] = b;
    }
}


template <typename T>
void checkResult(const thrust::host_vector<T> & pred, const thrust::host_vector<T> & gt)
{
    std::printf("result is %s\n", pred == gt ? "correct." : "WRONG!!!");

    #if false
    for (int i = 0; i < pred.size(); ++i)
    {
        printf("%f %f\n", pred[i], gt[i]);
    }
    #endif
}


int main(int argc, char * argv[])
{
    constexpr int n = 100'000;
    constexpr int biasSize = 10;
    float scale = 0.5f;
    constexpr int kDup = 1000;

    thrust::host_vector<uint8_t> h_mask(n);
    thrust::host_vector<float> h_add(n);
    thrust::host_vector<float> h_x(n);
    thrust::host_vector<float> h_y(n);
    thrust::host_vector<float> h_bias(biasSize);

    for (int i = 0; i < n; ++i)
    {
        h_mask[i] = static_cast<uint8_t>(i);
        h_add[i] = static_cast<float>(i);
        h_x[i] = static_cast<float>(i);
    }

    for (int i = 0; i < biasSize; ++i)
    {
        h_bias[i] = static_cast<float>(i);
    }

    thrust::host_vector<float> gt(n);

    for (int i = 0; i < n; ++i)
    {
        gt[i] = (h_x[i] + h_bias[i % biasSize]) * static_cast<float>(static_cast<bool>(h_mask[i]) * scale) + h_add[i];
    }

    thrust::device_vector<uint8_t> d_mask = h_mask;
    thrust::device_vector<float> d_add = h_add;
    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y(n);
    thrust::device_vector<float> d_bias = h_bias;

    MaskScaleAdd<float> msa(d_mask.data().get(), scale, d_add.data().get());

    constexpr dim3 block = {512};
    dim3 grid = {(n + block.x - 1) / block.x};

    float ms;
    cudaEvent_t ss, ee;
    CUDA_CHECK(cudaEventCreate(&ss));
    CUDA_CHECK(cudaEventCreate(&ee));


    // Naive.
    thrust::fill(d_y.begin(), d_y.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int i = 0; i < kDup; ++i)
    {
        fusedBiasedMaskScaleAddNaive<<<grid, block>>>(
            d_x.data().get(),
            n,
            d_bias.data().get(),
            biasSize,
            d_y.data().get(),
            msa
        );
    }

    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    h_y = d_y;
    std::printf("fusedBiasedMaskScaleAddNaive took %f ms, ", ms / kDup);
    checkResult(h_y, gt);


    // Smem.
    thrust::fill(d_y.begin(), d_y.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int i = 0; i < kDup; ++i)
    {
        fusedBiasedMaskScaleAddSmem<<<grid, block, biasSize>>>(
            d_x.data().get(),
            n,
            d_bias.data().get(),
            biasSize,
            d_y.data().get(),
            msa
        );
    }

    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    h_y = d_y;
    std::printf("fusedBiasedMaskScaleAddSmem took %f ms, ", ms / kDup);
    checkResult(h_y, gt);


    // Free cuda events.
    cudaEventDestroy(ss);
    cudaEventDestroy(ee);

    return EXIT_SUCCESS;
}
