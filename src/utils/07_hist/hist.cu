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


/// Each block accumulates number of >0-elements as smem,
/// and each thread knows its >0-element-offset in block (counter before accumulation).
/// Each block accumulates to a global offset, and the old offset is the begin position for this block to write to.
template <int kBlockSize>
__global__ void filterSmem(const int * __restrict__ in, int nx, int * __restrict__ out, int * __restrict__ ngtz)
{
    // 计数器声明为shared memory，去计数各个block范围内大于0的数量
    __shared__ int cnt;

    const int totalNumThreads = gridDim.x * kBlockSize;
    int tid = threadIdx.x;
    int from = blockIdx.x * blockDim.x + threadIdx.x;

    for ( ; from < nx; from += totalNumThreads)
    {
        // Zero the counter.
        if (tid == 0)
        {
            cnt = 0;
        }

        __syncthreads();

        int d = -1;
        int nonZeroOffsetInBlock;

        // cnt 表示每个 block 范围内大于0的数量，block 内的线程都可访问
        // pos 是每个线程私有的寄存器，
        // 且作为 atomicAdd 的返回值，表示当前线程对 cnt 原子加之前的 cnt，
        // 比如 1 2 4 号线程都大于 0，那么对于 4 号线程来说 cnt = 3, old = 2
        if (from < nx)
        {
            d = in[from];

            if (0 < d)
            {
                nonZeroOffsetInBlock = atomicAdd(&cnt, 1);
            }
        }

        __syncthreads();

        // Each block accumulates its numNonZero count to global ngtz.
        // The old ngtz is the first offset this thread should write.
        if (threadIdx.x == 0)
        {
            cnt = atomicAdd(ngtz, cnt);
        }

        __syncthreads();

        // Write & store.
        if (from < nx && 0 < d)
        {
            out[cnt + nonZeroOffsetInBlock] = d;
        }

        __syncthreads();
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

    thrust::device_vector<int> d_nres(1);
    thrust::device_vector<int> d_out = h_in;
    thrust::host_vector<int> h_nres(1);

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
    thrust::fill(thrust::device, d_nres.begin(), d_nres.end(), 0);
    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0);

    CUDA_CHECK(cudaEventRecord(ss));
    filterNaive<block.x><<<grid, block>>>(d_in.data().get(), n, d_out.data().get(), d_nres.data().get());
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));

    h_nres = d_nres;
    std::printf("filterNaive took %f ms, ", ms);
    checkResult(h_nres[0], gt);

    // Smem.
    thrust::fill(thrust::device, d_nres.begin(), d_nres.end(), 0);
    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0);
    CUDA_CHECK(cudaEventRecord(ss));
    filterSmem<block.x><<<grid, block>>>(d_in.data().get(), n, d_out.data().get(), d_nres.data().get());
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));

    h_nres = d_nres;
    std::printf("filterSmem took %f ms, ", ms);
    checkResult(h_nres[0], gt);

    // Warp.
    thrust::fill(thrust::device, d_nres.begin(), d_nres.end(), 0);
    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0);
    CUDA_CHECK(cudaEventRecord(ss));
    filterWarp<block.x><<<grid, block>>>(d_in.data().get(), n, d_out.data().get(), d_nres.data().get());
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));

    h_nres = d_nres;
    std::printf("filterWarp took %f ms, ", ms);
    checkResult(h_nres[0], gt);

    CUDA_CHECK(cudaEventDestroy(ss));
    CUDA_CHECK(cudaEventDestroy(ee));

    return EXIT_SUCCESS;
}

// filterNaive took 0.761440 ms, Result: 12800000 vs 12800000, is correct.
//
// filterSmem took 0.452768 ms, Result: 12800000 vs 12800000, is correct.
//
// filterWarp took 0.752672 ms, Result: 12800000 vs 12800000, is correct.
