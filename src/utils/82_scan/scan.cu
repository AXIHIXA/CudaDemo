#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


__device__ __forceinline__
void shfl_sync_up_b32(unsigned & d, unsigned & p, unsigned a, unsigned b, unsigned c, unsigned memberMask)
{
    asm volatile("{\n"
        ".reg .pred p;\n"
        "shfl.sync.up.b32 %0|p, %2, %3, %4, %5;\n"
        "selp.b32 %1, 1, 0, p;\n"
        "}\n"
            : "=r"(d), "=r"(p)
            : "r"(a), "r"(b), "r"(c), "r"(memberMask));
}


template <typename ScanOp, typename T, int kWarpThreads = 32>
__device__ void warpScan(int laneIdx, T & input, T & inclusiveOutput, T & exclusiveOutput, ScanOp & scanOp)
{
    static_assert(kWarpThreads == 32);

    inclusiveOutput = input;

    #pragma unroll
    for (int step = 1; step <= (kWarpThreads >> 1); step <<= 1)
    {
        // All lanes must actively participate in warp shuffle.
        // Needs lined PTX if splitting warps.
        T temp = __shfl_up_sync(0xffffffff, inclusiveOutput, step, kWarpThreads);

        // Only scan with a valid peer; do not scan with self.
        if (step <= laneIdx)
        {
            inclusiveOutput = scanOp(inclusiveOutput, temp);
        }
    }

    exclusiveOutput = __shfl_up_sync(0xffffffff, inclusiveOutput, 1, kWarpThreads);
}


template <int kBlockDimX, int kWarpThreads = 32, typename ScanOp, typename T>
__device__ void blockScan(T & input, T & inclusiveOutput, T & exclusiveOutput, T & blockAggregate, ScanOp scanOp)
{
    static_assert(kBlockDimX % kWarpThreads == 0 && kWarpThreads == 32);
    constexpr int kWarps = kBlockDimX / kWarpThreads;
    __shared__ T warpAggregate[kWarps];

    // Warp scan.
    const int tid = threadIdx.x;
    const int laneIdx = tid % kWarpThreads;
    const int warpIdx = tid / kWarpThreads;

    // Exclusive output for lane0s in all warps are invalid.
    warpScan(laneIdx, input, inclusiveOutput, exclusiveOutput, scanOp);

    // NOTE:
    // Last lane in warp, we DON't want "incomplete warps"!
    // Make block size a multiple of 32!
    if (laneIdx == kWarpThreads - 1)
    {
        warpAggregate[warpIdx] = inclusiveOutput;
    }

    __syncthreads();

    // Compute warp prefix.
    T warpPrefix = {};
    T regBlockAgg = warpAggregate[0];

    #pragma unroll
    for (int warp = 1; warp < kWarps; ++warp)
    {
        if (warpIdx == warp)
        {
            warpPrefix = regBlockAgg;
        }

        regBlockAgg = scanOp(regBlockAgg, warpAggregate[warp]);
    }

    if (tid == 0)
    {
        blockAggregate = regBlockAgg;
    }

    // Apply warp prefix.
    if (0 < warpIdx)
    {
        inclusiveOutput = scanOp(inclusiveOutput, warpPrefix);
        exclusiveOutput = laneIdx == 0 ? warpPrefix : scanOp(exclusiveOutput, warpPrefix);
    }
}


template <int kBlockDimX, int kThreadSpan, int kWarpThreads = 32, typename ScanOp, typename T>
__device__ void blockScan(T (& input)[kThreadSpan],
                          T (& inclusiveOutput)[kThreadSpan],
                          T & blockAggregate,
                          ScanOp scanOp)
{
    // Trivial case: One-element array.
    T sink;

    if constexpr (kThreadSpan == 1)
    {
        blockScan<kBlockDimX>(input[0], inclusiveOutput[0], sink, blockAggregate, scanOp);
        return;
    }

    // Thread aggregate by thread level reduction.
    T threadPrefix = input[0];

    #pragma unroll
    for (int i = 1; i < kThreadSpan; ++i)
    {
        threadPrefix = scanOp(threadPrefix, input[i]);
    }

    // Block level exclusive scan. Scans thread aggregates into thread prefixes.
    blockScan<kBlockDimX>(threadPrefix, sink, threadPrefix, blockAggregate, scanOp);

    // Thread level scan with prefixes.
    // Note that exclusive scan output for 0-th element is invalid.
    const int tid = threadIdx.x;

    inclusiveOutput[0] = 0 < tid ? scanOp(input[0], threadPrefix) : input[0];

    #pragma unroll
    for (int i = 1; i < kThreadSpan; ++i)
    {
        inclusiveOutput[i] = scanOp(inclusiveOutput[i - 1], input[i]);
    }

    #if false
    static_assert(kThreadSpan == 4);
    printf("tid %d input = %d %d %d %d prefix = %d output = %d %d %d %d\n",
           tid,
           input[0], input[1], input[2], input[3],
           threadPrefix,
           inclusiveOutput[0], inclusiveOutput[1], inclusiveOutput[2], inclusiveOutput[3]
    );
    #endif  // false
}


template <int kBlockDimX, int kThreadSpan, int kWarpThreads = 32, typename ScanOp, typename T>
__global__ void inclusiveScan(T * input,
                              T * inclusiveOutput,
                              int nx,
                              ScanOp scanOp)
{
    const int gx = blockIdx.x * kBlockDimX * kThreadSpan + threadIdx.x * kThreadSpan;
    T x[kThreadSpan];

    #pragma unroll
    for (int i = 0; i < kThreadSpan; ++i)
    {
        if (gx + i < nx)
        {
            x[i] = input[gx + i];
        }
    }

    T threadInclusiveOutput[kThreadSpan];
    T blockAggregate;
    blockScan<kBlockDimX, kThreadSpan>(x, threadInclusiveOutput, blockAggregate, scanOp);

    // TODO Intra-block blockAggregate scan on GMEM temp storage.

    #pragma unroll
    for (int i = 0; i < kThreadSpan; ++i)
    {
        if (gx + i < nx)
        {
            inclusiveOutput[gx + i] = threadInclusiveOutput[i];
        }
    }
}


int main(int argc, char * argv[])
{
    using T = int;
    const int nx = 2067;
    thrust::device_vector<T> devInput(nx, 1);
    thrust::device_vector<T> devInclusiveOutput(nx, 1);
    thrust::device_vector<T> devTempStorage(nx);

    constexpr int kThreadSpan = 4;
    constexpr int threadsNeeded = (nx + kThreadSpan - 1) / kThreadSpan;
    constexpr int nearestGreaterMultipleOf32 = (threadsNeeded + 31) / 32 * 32;
    constexpr dim3 kBlock(nearestGreaterMultipleOf32, 1);
    inclusiveScan<kBlock.x, kThreadSpan><<<1, kBlock>>>(
            thrust::raw_pointer_cast(devInput.data()),
            thrust::raw_pointer_cast(devInclusiveOutput.data()),
            nx,
            cub::Sum()
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::host_vector<T> hostInclusiveOutput = devInclusiveOutput;

    for (auto x : hostInclusiveOutput)
    {
        std::cout << x << ' ';
    }

    std::cout << '\n';

    return EXIT_SUCCESS;
}

// Commonly-used Nsight Compute metrics:
// Bank conflicts:
//     l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
//     l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
// Warp divergence:
//     smsp__thread_inst_executed_per_inst_executed
// L1/L2 cache hit rates:
//     l1tex__t_sector_hit_rate
//     lts__t_sector_hit_rate
// Global load/store throughputs and efficiencies:
//     l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second
//     l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second
//     smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
//     smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct
// Achieved and therotical occupancy:
//     sm__warps_active.avg.pct_of_peak_sustained_active
//     sm__maximum_warps_per_active_cycle_pct

/*
# Profile bank conflicts:
ncu -k regex:gemmSmem --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./cmake-build-release/demo
*/
