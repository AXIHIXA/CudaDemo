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


int filterCpu(const int * __restrict__ in, int nx, int * __restrict__ out)
{
    int ngtz = 0;

    for (int i = 0; i < nx; ++i)
    {
        if (0 < in[i])
        {
            out[ngtz++] = in[i];
        }
    }

    return ngtz;
}


template <int kBlockSize>
__global__ void filterNaive(const int * __restrict__ in, int nx, int * __restrict__ out, int * __restrict__ ngtz)
{
    const unsigned gridSize = gridDim.x;
    unsigned from = blockIdx.x * kBlockSize + threadIdx.x;

    for ( ; from < nx; from += gridSize * kBlockSize)
    {
        if (0 < in[from])
        {
            // Bottleneck: atomicAdd on global memory!
            out[atomicAdd(ngtz, 1)] = in[from];
        }
    }
}


/// Each block accumulates number of gt0-elements as smem,
/// and each thread knows its in-block element offset (counter before accumulation).
/// Each block accumulates to a global offset, and the old offset is the begin position for this block to write to.
template <int kBlockSize>
__global__ void filterSmem(const int * __restrict__ in, int nx, int * __restrict__ out, int * __restrict__ ngtz)
{
    // Block counter of greater-than-zero elements in this block.
    __shared__ int gtz;

    const int totalNumberOfThreads = gridDim.x * kBlockSize;

    int tid = threadIdx.x;
    int from = blockIdx.x * kBlockSize + threadIdx.x;

    // Grid translation.
    for ( ; from < nx; from += totalNumberOfThreads)
    {
        // Zero counter.
        if (tid == 0)
        {
            gtz = 0;
        }

        __syncthreads();

        // d: Set to actual value if in[from] > 0.
        // inBlockOffset: Offset of d (if > 0) in an array containing ONLY gt0 elements in this block.
        int d = -1;
        int inBlockOffset = 0;

        if (from < nx)
        {
            d = in[from];

            if (0 < d)
            {
                inBlockOffset = atomicAdd(&gtz, 1);
            }
        }

        __syncthreads();

        // Each block accumulates its number of gt0 elements to global counter.
        // Return value of atomicAdd indicates the begin position of this block to write to.
        // IMPORTANT: Reuse smem gtz counter (instead of register) to speed up.
        // Performance: smem gtz: 0.46ms; reg offset: 0.63ms!

        #if true
        if (tid == 0)
        {
            gtz = atomicAdd(ngtz, gtz);
        }

        __syncthreads();

        if (0 < d)
        {
            out[gtz + inBlockOffset] = d;
        }

        __syncthreads();
        #endif  // true

        #if false
        int ofst = 0;

        if (tid == 0)
        {
            ofst = atomicAdd(ngtz, gtz);
        }

        __syncthreads();

        if (0 < d)
        {
            out[ofst + inBlockOffset] = d;
        }

        __syncthreads();
        #endif  // false
    }
}


__device__ int atomicAggInc(int * ctr)
{
    unsigned active = __activemask();
    int leader = __ffs(active) - 1;  // Index of the first active thread in warp.
    int change = __popc(active);     // Number of active threads.

    // Inlines PTX assembly "mov.u32 laneMaskLt, %lanemask_lt;".
    // "=r" denotes laneMaskLt is an .u32 register which is written to.
    // "+r" denotes it's both read from and written to.
    // "%%" escapes "%" in raw PTX assembly.
    // https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
    int laneMaskLt;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(laneMaskLt));

    // Offset of this thread, in an array consisiting of ONLY active threads in this warp.
    unsigned rank = __popc(active & laneMaskLt);

    int warpRes;

    if (rank == 0)
    {
        // leader thread of every warp
        warpRes = atomicAdd(ctr, change);
    }

    // compute global offset of warp
    // broadcast warp_res of leader thread to every active thread
    warpRes = __shfl_sync(active, warpRes, leader);

    // global offset + local offset == final offset
    return warpRes + rank;
}


/// Atomic add at warp-register level, then each warp accumulate to global ngtz.
/// In-warp element offset is simply index of active warp (needs asm lanemask_lt).
template <int kBlockSize>
__global__ void filterWarp(const int * __restrict__ in, int nx, int * __restrict__ out, int * __restrict__ ngtz)
{
    int from = blockIdx.x * kBlockSize + threadIdx.x;

    if (nx <= from)
    {
        return;
    }

    if (0 < in[from])
    {
        // 过滤出 0 < in[from] 的线程，比如 warp #0 里面只有 thread #0 #1 满足，
        // 那么只有 thread #0 #1 运行，对应的 __activemask() 为 110000...00
        // atomicAggInc 计算当前 thread 负责数据的全局 offset
        out[atomicAggInc(ngtz)] = in[from];
    }
}


void checkResult(int res, int gt)
{
    std::printf("Result: %d vs %d, is %s\n\n", res, gt, res == gt ? "correct." : "WRONG!!!");
}


int main(int argc, char * argv[])
{
    constexpr int n = 25'600'000;
    thrust::host_vector<int> h_in(n);

    for (int i = 0; i < n; ++i)
    {
        h_in[i] = (i & 1) ? -1 : 1;
    }

    thrust::device_vector<int> d_in = h_in;
    int gt = std::count_if(h_in.cbegin(), h_in.cend(), [](int x) { return 0 < x; });

    thrust::device_vector<int> d_ngtz(1);
    thrust::device_vector<int> d_out = h_in;
    thrust::host_vector<int> h_ngtz(1);

    constexpr dim3 block = {256};
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp deviceProp = {};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    dim3 grid = {std::min<unsigned>((n + block.x - 1) / block.x, deviceProp.maxGridSize[0])};

    float ms;
    cudaEvent_t ss;
    cudaEvent_t ee;
    CUDA_CHECK(cudaEventCreate(&ss));
    CUDA_CHECK(cudaEventCreate(&ee));

    // Naive.
    thrust::fill(thrust::device, d_ngtz.begin(), d_ngtz.end(), 0);
    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0);

    CUDA_CHECK(cudaEventRecord(ss));
    filterNaive<block.x><<<grid, block>>>(d_in.data().get(), n, d_out.data().get(), d_ngtz.data().get());
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));

    h_ngtz = d_ngtz;
    std::printf("filterNaive took %f ms, ", ms);
    checkResult(h_ngtz[0], gt);

    // Smem.
    thrust::fill(thrust::device, d_ngtz.begin(), d_ngtz.end(), 0);
    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0);
    CUDA_CHECK(cudaEventRecord(ss));
    filterSmem<block.x><<<grid, block>>>(d_in.data().get(), n, d_out.data().get(), d_ngtz.data().get());
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));

    h_ngtz = d_ngtz;
    std::printf("filterSmem took %f ms, ", ms);
    checkResult(h_ngtz[0], gt);

    // Warp.
    thrust::fill(thrust::device, d_ngtz.begin(), d_ngtz.end(), 0);
    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0);
    CUDA_CHECK(cudaEventRecord(ss));
    filterWarp<block.x><<<grid, block>>>(d_in.data().get(), n, d_out.data().get(), d_ngtz.data().get());
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));

    h_ngtz = d_ngtz;
    std::printf("filterWarp took %f ms, ", ms);
    checkResult(h_ngtz[0], gt);

    CUDA_CHECK(cudaEventDestroy(ss));
    CUDA_CHECK(cudaEventDestroy(ee));

    return EXIT_SUCCESS;
}

// filterNaive took 0.761440 ms, Result: 12800000 vs 12800000, is correct.
//
// filterSmem took 0.452768 ms, Result: 12800000 vs 12800000, is correct.
//
// filterWarp took 0.752672 ms, Result: 12800000 vs 12800000, is correct.
