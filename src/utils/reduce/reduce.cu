#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
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
template <typename T>
__device__ void warpSmemReduce(volatile T * __restrict__ smem, unsigned tid)
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
    T x = smem[tid];

    if (64 <= blockDim.x)
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


template <int kBlockSize, typename T>
__global__ void reduceWithLastWarpUnrolled(const T * __restrict__ in, T * __restrict__ out)
{
    __shared__ T smem[kBlockSize];

    unsigned ix = blockIdx.x * kBlockSize * 2 + threadIdx.x;
    unsigned tid = threadIdx.x;

    smem[tid] = in[ix] + in[ix + kBlockSize];
    __syncthreads();

    for (unsigned s = (blockDim.x >> 1); 32 < s; s >>= 1)
    {
        if (tid < s)
        {
            smem[tid] += smem[tid + s];
        }

        __syncthreads();
    }

    // Reduce the last warp.
    if (tid < 32)
    {
        warpSmemReduce(smem, tid);
    }

    if (tid == 0)
    {
        out[blockIdx.x] = smem[0];
    }
}


template <int kBlockSize, typename T>
__device__ void blockSmemReduce(volatile T * __restrict__ smem)
{
    // Completely unroll the for loop,
    // wiping out addition instructions in for updates,
    // and offering compiler with more freedom for reordering.
    if (1024 <= kBlockSize)
    {
        if (threadIdx.x < 512)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 512];
        }

        __syncthreads();
    }

    if (512 <= kBlockSize)
    {
        if (threadIdx.x < 256)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 256];
        }

        __syncthreads();
    }

    if (256 <= kBlockSize)
    {
        if (threadIdx.x < 128)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 128];
        }

        __syncthreads();
    }

    if (128 <= kBlockSize)
    {
        if (threadIdx.x < 64)
        {
            smem[threadIdx.x] += smem[threadIdx.x + 64];
        }

        __syncthreads();
    }

    // The final warp.
    if (threadIdx.x < 32)
    {
        volatile float * vshm = smem;

        if (64 <= blockDim.x)
        {
            vshm[threadIdx.x] += vshm[threadIdx.x + 32];
        }

        vshm[threadIdx.x] += vshm[threadIdx.x + 16];
        vshm[threadIdx.x] += vshm[threadIdx.x + 8];
        vshm[threadIdx.x] += vshm[threadIdx.x + 4];
        vshm[threadIdx.x] += vshm[threadIdx.x + 2];
        vshm[threadIdx.x] += vshm[threadIdx.x + 1];
    }

}


template <int kBlockSize, typename T>
__global__ void reduceFullUnroll(const T * __restrict__ in, unsigned nx, T * __restrict__ out)
{
    __shared__ T smem[kBlockSize];

    unsigned ix = blockIdx.x * kBlockSize + threadIdx.x;
    unsigned tid = threadIdx.x;
    unsigned totalNumberOfthreads = kBlockSize * gridDim.x;

    T sum = 0;

    for (unsigned i = ix; i < nx; i += totalNumberOfthreads)
    {
        sum += in[i];
    }

    smem[tid] = sum;
    __syncthreads();

    blockSmemReduce<kBlockSize>(smem);

    if (tid == 0)
    {
        out[blockIdx.x] = smem[0];
    }
}


void checkResult(const thrust::device_vector<float> & in, const thrust::host_vector<float> & out)
{
    float res = thrust::reduce(out.cbegin(), out.cend());
    float gt = thrust::reduce(thrust::device, in.cbegin(), in.cend());
    std::printf("Result: %f vs %f, is %s\n\n", res, gt, res == gt ? "correct." : "WRONG!!!");
}


int main(int argc, char * argv[])
{
    constexpr int n = 25'600'000;
    thrust::host_vector<float> h_in(n, 1.0f);
    thrust::device_vector<float> d_in = h_in;
    thrust::device_vector<float> d_out(n, 0.0f);
    thrust::device_vector<float> d_part_out(n, 0.0f);
    thrust::host_vector<float> h_out;

    #ifdef NDEBUG
    constexpr int kDup = 100;
    #else
    constexpr int kDup = 1;
    #endif  // NDEBUG

    constexpr dim3 block = {256};
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

    // Last warp as unrolled warp-wide reduction.
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

    // Unrolled block-wide reduction.
    #ifdef NDEBUG
    reduceFullUnroll<block.x><<<grid, block>>>(
            d_in.data().get(), n, d_part_out.data().get()
    );
    reduceFullUnroll<block.x><<<1, block>>>(
            d_part_out.data().get(), grid.x, d_out.data().get()
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    #endif  // NDEBUG

    thrust::fill(thrust::device, d_part_out.begin(), d_part_out.end(), 0.0f);
    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int dup = 0; dup < kDup; ++dup)
    {
        reduceFullUnroll<block.x><<<grid, block>>>(
                d_in.data().get(), n, d_part_out.data().get()
        );
        reduceFullUnroll<block.x><<<1, block>>>(
                d_part_out.data().get(), grid.x, d_out.data().get()
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

    cudaEventDestroy(ss);
    cudaEventDestroy(ee);

    return EXIT_SUCCESS;
}
