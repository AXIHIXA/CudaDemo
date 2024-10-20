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
__global__ void reduceFatBlock(const T * __restrict__ in, T * __restrict__ out)
{
    __shared__ T smem[kBlockSize];

    unsigned ix = blockIdx.x * kBlockSize * 2 + threadIdx.x;
    unsigned tid = threadIdx.x;

    smem[tid] = in[ix] + in[ix + kBlockSize];
    __syncthreads();

    // We could simply floor-divide by 2 here, as this operates smem (of size kBlockSize).
    // Safe as long as kBlockSize is a power of 2 (which, in our case, is 256).
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


/// Recommended way to implement warp-wide reduction by NVIDIA.
/// Ensure load/store order and result correctness by synchronization primitives.
template <unsigned kBlockSize, typename T>
__device__ void warpSmemReduce(volatile T * __restrict__ smem)
{
    // There's risk of data race (and writes/loads being optimized out) on smem:
    // E.g., smem[tid] += smem[tid + 16] ==> t0: smem[0] += smem[16]; t16: smem[16] += smem[32].
    // Thread #0 and #16 will race on smem[16].
    // We introduce an intermediate register (with  __syncwarp intrinsic and volatile smem) to wipe out data race.
    //
    // Notes on volatile (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#volatile-qualifier):
    //
    // The compiler is free to optimize reads and writes to global or shared memory
    // (for example, by caching global reads into registers or L1 cache)
    // as long as it respects the memory ordering semantics of memory fence functions
    // and memory visibility semantics of synchronization functions.
    //
    // These optimizations can be disabled using the volatile keyword:
    // If a variable located in global or shared memory is declared as volatile,
    // the compiler assumes that its value can be changed or used at any time by another thread
    // and therefore any reference to this variable compiles to an actual memory read or write instruction.
    unsigned tid = threadIdx.x;
    T x = smem[tid];

    if (64 <= kBlockSize)
    {
        x += smem[tid + 32]; __syncwarp();
        smem[tid] = x; __syncwarp();
    }

    x += smem[tid + 16]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 8]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 4]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 2]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 1]; __syncwarp();
    smem[tid] = x; __syncwarp();
}


template <unsigned kBlockSize, typename T>
__global__ void reduceWithLastWarpUnrolled(const T * __restrict__ in, T * __restrict__ out)
{
    __shared__ T smem[kBlockSize];

    unsigned tid = threadIdx.x;
    unsigned from = blockIdx.x * kBlockSize * 2 + threadIdx.x;

    smem[tid] = in[from] + in[from + kBlockSize];
    __syncthreads();

    for (unsigned s = kBlockSize >> 1; 32 < s; s >>= 1)
    {
        if (tid < s)
        {
            smem[tid] += smem[tid + s];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        warpSmemReduce<kBlockSize>(smem);
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


template <int kBlockSize, typename T>
__global__ void reduceFullUnroll(const T * __restrict__ in, T * __restrict__ out)
{
    __shared__ T smem[kBlockSize];

    int from = blockIdx.x * kBlockSize * 2 + threadIdx.x;
    int tid = threadIdx.x;

    smem[tid] = in[from] + in[from + kBlockSize];
    __syncthreads();

    blockSmemReduce<kBlockSize>(smem);

    if (tid == 0)
    {
        out[blockIdx.x] = smem[0];
    }
}


// Use grid translation (adapt to input size) instead of block translation (of compile-time constant steps).
// Adapts to input size automatically, at a cost of degraded performance.
// Trade-off.
template <int kBlockDimX, typename T, int kWarpThreads = 32>
__global__ void reduceGridTranslationFullUnroll(const T * __restrict__ src, int nx, T * __restrict__ dst)
{
    static_assert(kWarpThreads == 32 && kBlockDimX % kWarpThreads == 0);

    const int tid = threadIdx.x;
    const int bbx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    __shared__ T smem[kBlockDimX];

    smem[tid] = 0;

    for (int gx = bbx; gx < nx; gx += stride)
    {
        smem[tid] += src[gx];
    }

    __syncthreads();

    #pragma unroll
    for (int step = (kBlockDimX >> 1); kWarpThreads < step; step >>= 1)
    {
        if (tid < step)
        {
            smem[tid] += smem[tid + step];
        }

        __syncthreads();
    }

    if (tid < kWarpThreads)
    {
        volatile T * vshm = smem;
        T x = vshm[tid];

        if (64 <= kBlockDimX)
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

    if (tid == 0)
    {
        dst[blockIdx.x] = smem[0];
    }
}


template <typename T>
__device__ T warpReduce(T val)
{
    constexpr unsigned mask = 0xffffffff;
    constexpr int kWarpThreads = 32;

    #if false
    // Does not populate warp reduction result to all threads.
    // Sub-optimal to the butterfly variant below.
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
    #endif  // false

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#reduction-across-a-warp
    #pragma unroll
    for (int laneMask = (kWarpThreads >> 1); 0 < laneMask; laneMask >>= 1)
    {
        val += __shfl_xor_sync(mask, val, laneMask, kWarpThreads);
    }

    return val;
}


/// Reach best performance when input size is around 25'600.
/// Degrades to sub-optimal when input size grows larger.
template <unsigned kBlockSize, unsigned kWarpSize, typename T>
__global__ void reduceGridTranslationWarpShuffle(const T * __restrict__ in, int nx, T * __restrict__ out)
{
    const unsigned gridSize = gridDim.x;
    unsigned from = blockIdx.x * kBlockSize + threadIdx.x;
    unsigned tid = threadIdx.x;

    T sum = 0;

    for (; from < nx; from += gridSize * kBlockSize)
    {
        sum += in[from];
    }

    __shared__ T warpSums[kBlockSize / kWarpSize];

    const int laneIdx = tid % kWarpSize;
    const int warpIdx = tid / kWarpSize;

    sum = warpReduce(sum);

    if (laneIdx == 0)
    {
        warpSums[warpIdx] = sum;
    }

    __syncthreads();

    sum = (tid < kBlockSize / kWarpSize) ? warpSums[laneIdx] : 0;

    // Final reduce using the first warp.
    // We know that each block (of size 256) will have 8 warps only.
    // Thus a second warpShuffleReduce will sum the 8 warpSums in this block.
    if (warpIdx == 0)
    {
        sum = warpReduce(sum);
    }

    if (tid == 0)
    {
        out[blockIdx.x] = sum;
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
    constexpr int kDup = 100;
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

    // Fat block
    #ifdef NDEBUG
    reduceFatBlock<(block.x >> 1)><<<grid, block.x >> 1>>>(
        d_in.data().get(), d_out.data().get()
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    #endif  // NDEBUG

    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int dup = 0; dup < kDup; ++dup)
    {
        reduceFatBlock<(block.x >> 1)><<<grid, block.x >> 1>>>(
            d_in.data().get(), d_out.data().get()
        );
    }

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));

    h_out = d_out;
    std::printf("reduceFatBlock: ");
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    std::printf("took %f ms, ", ms / kDup);
    checkResult(d_in, h_out);

    // Unrolled last warp.
    #ifdef NDEBUG
    reduceWithLastWarpUnrolled<(block.x >> 1)><<<grid, block.x >> 1>>>(
        d_in.data().get(), d_out.data().get()
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    #endif  // NDEBUG

    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int dup = 0; dup < kDup; ++dup)
    {
        reduceWithLastWarpUnrolled<(block.x >> 1)><<<grid, block.x >> 1>>>(
            d_in.data().get(), d_out.data().get()
        );
    }

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));

    h_out = d_out;
    std::printf("reduceWithLastWarpUnrolled: ");
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    std::printf("took %f ms, ", ms / kDup);
    checkResult(d_in, h_out);

    // Full unroll.
    #ifdef NDEBUG
    reduceFullUnroll<(block.x >> 1)><<<grid, block.x >> 1>>>(
        d_in.data().get(), d_out.data().get()
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    #endif  // NDEBUG

    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int dup = 0; dup < kDup; ++dup)
    {
        reduceFullUnroll<(block.x >> 1)><<<grid, block.x >> 1>>>(
            d_in.data().get(), d_out.data().get()
        );
    }

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));

    h_out = d_out;
    std::printf("reduceFullUnroll: ");
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    std::printf("took %f ms, ", ms / kDup);
    checkResult(d_in, h_out);

    // Grid-translation to adapt input size.
    // Last warp as unrolled warp-wide reduction.
    // Degraded performance (unevitable cost).
    #ifdef NDEBUG
    reduceGridTranslationFullUnroll<block.x><<<grid, block.x>>>(
        d_in.data().get(), n, d_out.data().get()
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    #endif  // NDEBUG

    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int dup = 0; dup < kDup; ++dup)
    {
        reduceGridTranslationFullUnroll<block.x><<<grid, block.x>>>(
            d_in.data().get(), n, d_out.data().get()
        );
    }

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));

    h_out = d_out;
    std::printf("reduceGridTranslationFullUnroll: ");
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    std::printf("took %f ms, ", ms / kDup);
    checkResult(d_in, h_out);

    // Warp shuffle reduction.
    #ifdef NDEBUG
    reduceGridTranslationWarpShuffle<block.x, 32><<<grid, block.x>>>(
        d_in.data().get(), n, d_out.data().get()
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    #endif  // NDEBUG

    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int dup = 0; dup < kDup; ++dup)
    {
        reduceGridTranslationWarpShuffle<block.x, 32><<<grid, block.x>>>(
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
