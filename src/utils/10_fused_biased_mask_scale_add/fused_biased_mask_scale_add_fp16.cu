#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


/// (x + bias) * mask * scale + add


template <typename T>
struct MaskScaleAdd
{
    MaskScaleAdd(const uint8_t * __restrict__ mask,
                 float scale,
                 const T * __restrict__ add) :
        mask(mask),
        scale(scale),
        add(add)
    {

    }

    __device__ T operator()(T x, int i) const
    {
        return x * static_cast<T>(static_cast<bool>(mask[i]) * scale) + add[i];
    }

    const uint8_t * __restrict__ mask = nullptr;
    float scale = 0.0f;
    const T * __restrict__ add = nullptr;
};


template <>
struct MaskScaleAdd<half>
{
    MaskScaleAdd(const uint8_t * __restrict__ mask,
                 float scale,
                 const half * __restrict__ add) :
        mask(mask),
        scale(scale),
        add(add)
    {

    }

    __device__ half operator()(half x, int i) const
    {
        return x * static_cast<half>(static_cast<bool>(mask[i]) * scale) + add[i];
    }

    /// Vectorized load and computation for half type
    /// (unlike float, which does not support vectorized computation.)
    __device__ half2 operator()(half2 x, int64_t i) const
    {
        uchar2 c_mask_i = reinterpret_cast<const uchar2 * __restrict__>(mask)[i];
        half2 add_i = reinterpret_cast<const half2 * __restrict__>(add)[i];

        half2 mask_i;
        mask_i.x = (c_mask_i.x != 0);
        mask_i.y = (c_mask_i.y != 0);

        #if false
        if (i == 1)
        {
            printf("x = %f %f mask = %f %f scale = %f add = %f %f\n",
                   __half2float(x.x), __half2float(x.y),
                   __half2float(mask_i.x), __half2float(mask_i.y),
                   scale,
                   __half2float(add_i.x), __half2float(add_i.y));
        }
        #endif

        return __hadd2(__hmul2(__hmul2(x, mask_i), __float2half2_rn(scale)), add_i);
    }

    const uint8_t * __restrict__ mask = nullptr;
    float scale = 0.0f;
    const half * __restrict__ add = nullptr;
};


template <typename Func>
__global__ void fusedBiasedMaskScaleAdd(const half * __restrict__ x,
                                        int nx,
                                        const half * __restrict__ glbBias,
                                        int biasSize,
                                        half * __restrict__ y,
                                        Func func)
{
    extern __shared__ half2 bias[];  // (biasSize >> 1,)
    biasSize >>= 1;

    const int stride = gridDim.x * blockDim.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // bias is a small size and loaded for multiple times.
    // Move from gmem to smem.
    if (tid < biasSize)
    {
        bias[tid] = reinterpret_cast<const half2 * __restrict__>(glbBias)[tid];
    }

    __syncthreads();

    for (int i = gid ; i < (nx >> 1); i += stride)
    {
        half2 a = __hadd2(reinterpret_cast<const half2 * __restrict__>(x)[i],
                          reinterpret_cast<const half2 * __restrict__>(bias)[i % biasSize]);
        reinterpret_cast<half2 * __restrict__>(y)[i] = func(a, i);
    }
}


template <typename U, typename V>
void checkResult(const thrust::host_vector<U> & pred, const thrust::host_vector<V> & gt)
{
    bool correct = true;

    for (int i = 0; i < gt.size(); ++i)
    {
        if (static_cast<V>(pred[i]) != gt[i])
        {
            correct = false;
            break;
        }
    }

    std::printf("result is %s\n", correct ? "correct." : "WRONG!!!");

    #if false
    for (int i = 0; i < pred.size(); ++i)
    {
        printf("%f %f\n", static_cast<float>(pred[i]), static_cast<float>(gt[i]));
    }
    #endif
}


int main(int argc, char * argv[])
{
    constexpr int n = 100;
    constexpr int biasSize = 10;
    float scale = 0.5f;
    constexpr int kDup = 1;

    thrust::host_vector<uint8_t> h_mask(n);
    thrust::host_vector<half> h_add(n);
    thrust::host_vector<half> h_x(n);
    thrust::host_vector<half> h_y(n);
    thrust::host_vector<half> h_bias(biasSize);

    for (int i = 0; i < n; ++i)
    {
        h_mask[i] = static_cast<uint8_t>(i);
        h_add[i] = static_cast<half>(i);
        h_x[i] = static_cast<half>(i);
    }

    for (int i = 0; i < biasSize; ++i)
    {
        h_bias[i] = static_cast<half>(i);
    }

    thrust::host_vector<float> gt(n);

    for (int i = 0; i < n; ++i)
    {
        gt[i] = (static_cast<float>(h_x[i]) + static_cast<float>(h_bias[i % biasSize])) *
                static_cast<float>(static_cast<bool>(h_mask[i]) * scale) + static_cast<float>(h_add[i]);
    }

    thrust::device_vector<uint8_t> d_mask = h_mask;
    thrust::device_vector<half> d_add = h_add;
    thrust::device_vector<half> d_x = h_x;
    thrust::device_vector<half> d_y(n);
    thrust::device_vector<half> d_bias = h_bias;

    MaskScaleAdd<half> msa(d_mask.data().get(), scale, d_add.data().get());

    constexpr dim3 block = {512};
    dim3 grid = {(n + block.x - 1) / block.x};

    float ms;
    cudaEvent_t ss, ee;
    CUDA_CHECK(cudaEventCreate(&ss));
    CUDA_CHECK(cudaEventCreate(&ee));

    // Test.
    thrust::fill(d_y.begin(), d_y.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int i = 0; i < kDup; ++i)
    {
        fusedBiasedMaskScaleAdd<<<grid, block, biasSize >> 1>>>(
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
    std::printf("fusedBiasedMaskScaleAdd took %f ms, ", ms / kDup);
    checkResult(h_y, gt);


    // Free cuda events.
    cudaEventDestroy(ss);
    cudaEventDestroy(ee);

    return EXIT_SUCCESS;
}
