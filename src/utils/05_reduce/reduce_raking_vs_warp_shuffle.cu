#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


/// Safe as long as total length of data is a multiple of kBlockSize.
/// Does not need to be a power of 2.
///
/// Note: using blockSize as a template arg can benefit from NVCC compiler optimization,
/// which is better than using blockDim.x that is known in runtime.
template <int kBlockSize, typename T>
__global__ void reduceNaive(const T * __restrict__ in, T * __restrict__ out)
{
    __shared__ T smem[kBlockSize];

    unsigned ix = blockIdx.x * kBlockSize + threadIdx.x;
    unsigned tid = threadIdx.x;

    smem[tid] = in[ix];
    __syncthreads();

    // Diverges when s < 32.
    // Optimizes this issue in the next version.
    for (unsigned s = (blockDim.x >> 1); 0 < s; s >>= 1)
    {
        if (tid < s)
        {
            smem[tid] += smem[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = smem[0];
    }
}


template <int kBlockSize, typename T>
__device__ void blockSmemReduce(T * __restrict__ smem)
{
    int tid = threadIdx.x;

    // Reference of the unroll below.
    #if false
    #pragma unroll
    for (unsigned s = (kBlockSize >> 1); 32 < s; s >>= 1)
    {
        if (tid < s)
        {
            smem[tid] += smem[tid + s];
        }

        __syncthreads();
    }
    #endif  // false

    // Completely unroll the for loop,
    // wiping out addition instructions in for updates,
    // and offering compiler with more freedom for reordering.
    // Note: Hardware constraint: blockSize is typically at most 1024.
    if (1024 <= kBlockSize)
    {
        if (tid < 512)
        {
            smem[tid] += smem[tid + 512];
        }

        __syncthreads();
    }

    if (512 <= kBlockSize)
    {
        if (tid < 256)
        {
            smem[tid] += smem[tid + 256];
        }

        __syncthreads();
    }

    if (256 <= kBlockSize)
    {
        if (tid < 128)
        {
            smem[tid] += smem[tid + 128];
        }

        __syncthreads();
    }

    if (128 <= kBlockSize)
    {
        if (tid < 64)
        {
            smem[tid] += smem[tid + 64];
        }

        __syncthreads();
    }

    // The final warp unrolled.
    if (tid < 32)
    {
        volatile T * __restrict__ vshm = smem;
        T x = vshm[tid];

        // hzw demoed that not explicitly using intermediate register x
        // and not calling __syncwarp() might also work correctly on pre-Volta GPUs.
        // However, this following is recommended by NVIDIA and is guaranteed to be correct!
        if (64 <= blockDim.x)
        {
            x += vshm[tid + 32]; __syncwarp();
            vshm[tid] = x; __syncwarp();
        }

        x += vshm[tid + 16]; __syncwarp();
        vshm[tid] = x; __syncwarp();
        x += vshm[tid + 8]; __syncwarp();
        vshm[tid] = x; __syncwarp();
        x += vshm[tid + 4]; __syncwarp();
        vshm[tid] = x; __syncwarp();
        x += vshm[tid + 2]; __syncwarp();
        vshm[tid] = x; __syncwarp();
        x += vshm[tid + 1]; __syncwarp();
        vshm[tid] = x; __syncwarp();
    }
}


// Use grid translation (adapt to input size) instead of block translation (of compile-time constant steps).
// Adapts to input size automatically, at a cost of degraded performance.
// Trade-off.
template <int kBlockDimX, typename T>
__global__ void reduceGridTranslationRaking(const T * __restrict__ src, const int nx, T * __restrict__ dst)
{
    const int tid = static_cast<int>(threadIdx.x);
    const int offset = blockIdx.x * kBlockDimX;
    const int stride = gridDim.x * kBlockDimX;

    T x = 0;

    for (int gx = offset + tid; gx < nx; gx += stride)
    {
        x += src[gx];
    }

    __shared__ T smem[kBlockDimX];
    smem[tid] = x;
    __syncthreads();

    #pragma unroll
    for (int s = (kBlockDimX >> 1); 0 != s; s >>= 1)
    {
        if (tid < s)
        {
            smem[tid] += smem[tid + s];
        }

        __syncthreads();
    }

    if (0 == tid)
    {
        dst[blockIdx.x] = smem[0];
    }
}


template <typename ReductionOp, typename T, int kWarpSize = 32>
__device__ T warpReduce(T x)
{
    ReductionOp op;

    #pragma unroll
    for (int s = (kWarpSize >> 1); 0 != s; s >>= 1)
    {
        x = op(x, __shfl_xor_sync(~0u, x, s, kWarpSize));
    }

    return x;
}


/// Reach best performance when input size is around 25'600.
/// Degrades to sub-optimal when input size grows larger.
template <int kBlockDimX, typename T, int kWarpSize = 32>
__global__ void reduceGridTranslationWarpShuffle(const T * __restrict__ src, const int nx, T * __restrict__ dst)
{
    const int tid = static_cast<int>(threadIdx.x);
    const int offset = blockIdx.x * kBlockDimX;
    const int stride = gridDim.x * kBlockDimX;

    T x = 0;

    for (int gx = offset + tid; gx < nx; gx += stride)
    {
        x += src[gx];
    }

    x = warpReduce<cub::Sum>(x);

    const int laneIdx = tid % kWarpSize;
    const int warpIdx = tid / kWarpSize;
    constexpr int kNumWarps = kBlockDimX / kWarpSize;

    __shared__ T smem[kNumWarps];

    if (0 == laneIdx)
    {
        smem[warpIdx] = x;
    }

    __syncthreads();

    if (0 == warpIdx)
    {
        x = laneIdx < kNumWarps ? smem[laneIdx] : 0;
        x = warpReduce<cub::Sum>(x);
    }

    if (0 == tid)
    {
        dst[blockIdx.x] = x;
    }
}


void checkResult(const thrust::device_vector<float> & in, const thrust::host_vector<float> & out)
{
    constexpr float kRTol = 2e-4f;
    constexpr float kATol = 1e-4f;

    float res = thrust::reduce(out.cbegin(), out.cend());
    float gt = thrust::reduce(thrust::device, in.cbegin(), in.cend());
    bool allclose = std::abs(res - gt) <= kATol + kRTol * std::abs(gt);
    std::printf("Result: %f vs %f, is %s\n\n", res, gt, allclose ? "correct." : "WRONG!!!");
}


int main(int argc, char * argv[])
{
    constexpr int n = 25'600'000;
    thrust::host_vector<float> h_in(n, 1.0f);
    thrust::host_vector<float> h_out;

    unsigned seed = std::random_device()();
    std::default_random_engine e(seed);
    std::uniform_real_distribution<float> d(1, 1);  // Setting to (1, 2) exceeds float precision!
    auto g = [&d, &e]() { return d(e); };
    std::generate(h_in.begin(), h_in.end(), g);

    thrust::device_vector<float> d_in = h_in;
    thrust::device_vector<float> d_out(n, 0.0f);

    #ifdef NDEBUG
    constexpr int kDup = 1;
    #else
    constexpr int kDup = 1;
    #endif  // NDEBUG

    constexpr dim3 block(256);
    dim3 grid = {(n + block.x - 1) / block.x};

    float ms;
    cudaEvent_t ss, ee;
    CUDA_CHECK(cudaEventCreate(&ss));
    CUDA_CHECK(cudaEventCreate(&ee));

    // Naive
    #ifdef NDEBUG
    reduceNaive<block.x><<<grid, block>>>(
            d_in.data().get(), d_out.data().get()
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    #endif  // NDEBUG

    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int dup = 0; dup < kDup; ++dup)
    {
        reduceNaive<block.x><<<grid, block>>>(
                d_in.data().get(), d_out.data().get()
        );
    }

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));

    h_out = d_out;
    std::printf("reduceNaive: ");
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    std::printf("took %f ms, ", ms / kDup);
    checkResult(d_in, h_out);

    // Raking-style block reduction with smem.
    #ifdef NDEBUG
    reduceGridTranslationRaking<block.x><<<grid, block.x>>>(
        d_in.data().get(), n, d_out.data().get()
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    #endif  // NDEBUG

    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int dup = 0; dup < kDup; ++dup)
    {
        reduceGridTranslationRaking<block.x><<<grid, block.x>>>(
            d_in.data().get(), n, d_out.data().get()
        );
    }

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));

    h_out = d_out;
    std::printf("reduceGridTranslationRaking: ");
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    std::printf("took %f ms, ", ms / kDup);
    checkResult(d_in, h_out);

    // Warp shuffle reduction.
    #ifdef NDEBUG
    reduceGridTranslationWarpShuffle<block.x><<<grid, block.x>>>(
        d_in.data().get(), n, d_out.data().get()
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    #endif  // NDEBUG

    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int dup = 0; dup < kDup; ++dup)
    {
        reduceGridTranslationWarpShuffle<block.x><<<grid, block.x>>>(
            d_in.data().get(), n, d_out.data().get()
        );
    }

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));

    h_out = d_out;
    std::printf("reduceGridTranslationWarpShuffle: ");
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    std::printf("took %f ms, ", ms / kDup);
    checkResult(d_in, h_out);

    // Free cuda events.
    cudaEventDestroy(ss);
    cudaEventDestroy(ee);

    return EXIT_SUCCESS;
}

/*
reduceNaive: took 0.646083 ms, Result: 25600000.000000 vs 25600000.000000, is correct.

reduceFatBlock: took 0.295687 ms, Result: 25600000.000000 vs 25600000.000000, is correct.

reduceWithLastWarpUnrolled: took 0.181950 ms, Result: 25600000.000000 vs 25600000.000000, is correct.

reduceFullUnroll: took 0.180863 ms, Result: 25600000.000000 vs 25600000.000000, is correct.

reduceGridTranslationFullUnroll: took 0.364220 ms, Result: 25600000.000000 vs 25600000.000000, is correct.

reduceGridTranslationWarpShuffle: took 0.291430 ms, Result: 25600000.000000 vs 25600000.000000, is correct.
*/
