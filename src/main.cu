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


static constexpr int kWarpSize = 32;


/// Softmax on innermost dimension.
void cpuSoftmax(const float * __restrict__ in, int nx, int ny, float * __restrict__ out)
{
    for (int j = 0; j < ny; ++j)
    {
        float sigma = 0.0f;
        float maxi = 0.0f;

        for (int i = 0; i < nx; ++i)
        {
            maxi = std::max(in[j * nx + i], maxi);
        }

        for (int i = 0; i < nx; ++i)
        {
            sigma += std::exp(in[j * nx + i] - maxi);
        }

        for (int i = 0; i < nx; ++i)
        {
            out[j * nx + i] = std::exp(in[j * nx + i] - maxi) / sigma;
        }
    }
}


template <typename T, int kVecSize>
struct alignas(sizeof(T) * kVecSize) Vec
{
    T val[kVecSize];
};

template <typename T>
struct Sum
{
    __device__ __forceinline__ T operator()(const T & a, const T & b) const
    {
        return a + b;
    }
};

template <typename T>
struct Max
{
    __device__ __forceinline__ T operator()(const T & a, const T & b) const
    {
        return max(a, b);
    }
};


template <template <typename> class ReductionOp, typename T, int kWarpWidth = kWarpSize>
__inline__ __device__ T warpReduce(T val)
{
    for (int mask = kWarpWidth >> 1; 0 < mask; mask >>= 1)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }

    return val;
}


void checkResult(const float * __restrict__ pred, const float * __restrict__ gt, int nx, int ny)
{
    bool correct = true;

    for (int i = 0; i < nx * ny; ++i)
    {
        if (pred[i] != gt[i])
        {
            correct = false;
            break;
        }
    }

    std::printf("result is %s\n", correct ? "correct." : "WRONG!!!");

    #if false
    for (int i = 0; i < nx * ny; ++i)
    {
        printf("%f %f\n", pred[i], gt[i]);
    }
    #endif
}


int main(int argc, char * argv[])
{
    constexpr int n = 1'024'000;
    constexpr int biasSize = 10;
    constexpr int kDup = 1;

    thrust::host_vector<float> h_x(n, 1.0f);
    thrust::host_vector<float> gt(n);


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
        softmax<<<grid, block, biasSize >> 1>>>(
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
    std::printf("softmax took %f ms, ", ms / kDup);
    checkResult(h_y, gt);


    // Free cuda events.
    cudaEventDestroy(ss);
    cudaEventDestroy(ee);

    return EXIT_SUCCESS;
}
