#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


__global__ void test1()
{
    __shared__ float smem[32 * 4];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float4 reg = {1.0f, 2.0f, 3.0f, 4.0f};
    auto smem4 = reinterpret_cast<float4 *>(smem);
    smem4[tid] = reg;
    __trap();
    __syncthreads();
}


__global__ void test2()
{
    __shared__ float smem[32 * 4];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    auto smem4 = reinterpret_cast<float4 *>(smem);
    smem4[tid] = {1.0f, 2.0f, 3.0f, 4.0f};
    __syncthreads();
}


__global__ void test3()
{
    __shared__ float smem[32 * 4];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    smem[tid * 4] = 1.0f;
    smem[tid * 4 + 1] = 2.0f;
    smem[tid * 4 + 2] = 3.0f;
    smem[tid * 4 + 3] = 4.0f;
    __syncthreads();
}


int main(int argc, char * argv[])
{
    test1<<<1, 32>>>();
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

//    test2<<<1, 32>>>();
//    CUDA_CHECK_LAST_ERROR();
//    CUDA_CHECK(cudaDeviceSynchronize());
//
//    test3<<<1, 32>>>();
//    CUDA_CHECK_LAST_ERROR();
//    CUDA_CHECK(cudaDeviceSynchronize());

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
