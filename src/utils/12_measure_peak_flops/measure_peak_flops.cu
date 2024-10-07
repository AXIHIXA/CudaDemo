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
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


template <int kDups>
__global__ void fp32Flops(const float * __restrict__ x,
                          const float * __restrict__ y,
                          float * __restrict__ result,
                          int * __restrict__ start,
                          int * __restrict__ stop)
{
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;

    float d1 = x[gtid];
    float d2 = y[gtid];

    float res = 0.0f;

    // Only measure the computation time, eliminate the memory access time.
    int startClock = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(startClock)::"memory");

    // Q1: Why use 4 fma instructions to get GPU peak performance?
    // A1: We use 2+ fma instructions to hide for loop comparsion and addition instruction overhead.
    // Q2: Why use 4 dependant fma instructions to get GPU peak performance, can we use 4 independant ones?
    // A2: Yes.
    for (int i = 0; i < kDups; ++i)
    {
        asm volatile(
                "{ \n\t"
                "fma.rn.f32 %0, %1, %2, %0; \n\t"
                "fma.rn.f32 %0, %1, %2, %0; \n\t"
                "fma.rn.f32 %0, %1, %2, %0; \n\t"
                "fma.rn.f32 %0, %1, %2, %0; \n\t"
                "} \n\t"
                :
                "+f"(res),
                "+f"(d1),
                "+f"(d2)
        );

        #if false
        // The inlined PTX is equivalent to:
        res = d1 * d2 + res;
        res = d1 * d2 + res;
        res = d1 * d2 + res;
        res = d1 * d2 + res;
        #endif  // false
    }

    // Sync all threads.
    asm volatile("bar.sync 0;");

    int stopClock = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stopClock)::"memory");

    result[gtid] = res;
    start[gtid] = startClock;
    stop[gtid] = stopClock;
}


int main(int argc, char * argv[])
{
    constexpr int kArraySize = 1024;
    constexpr int kMaxThreadsPerSM = 1024;
    constexpr int kDups = 1000;

    thrust::host_vector<float> hostA(kArraySize);
    thrust::host_vector<float> hostB(kArraySize);
    std::iota(hostA.begin(), hostA.end(), 0.0f);
    std::iota(hostB.begin(), hostB.end(), 0.0f);

    thrust::device_vector<float> devA = hostA;
    thrust::device_vector<float> devB = hostB;
    thrust::device_vector<int> devStartClock(kMaxThreadsPerSM);
    thrust::device_vector<int> devStopClock(kMaxThreadsPerSM);
    thrust::device_vector<float> devResult(1);

    // Confirm launch max threads of SM == 1024 to do FMA to saturate SM resource.
    fp32Flops<kDups><<<1, kMaxThreadsPerSM>>>(
            thrust::raw_pointer_cast(devA.data()),
            thrust::raw_pointer_cast(devB.data()),
            thrust::raw_pointer_cast(devResult.data()),
            thrust::raw_pointer_cast(devStartClock.data()),
            thrust::raw_pointer_cast(devStopClock.data())
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::host_vector<int> hostStartClock = devStartClock;
    thrust::host_vector<int> hostStopClock = devStopClock;

    cudaDeviceProp props = {};
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    float flops = (kDups * 4 * 2 * kArraySize) / (static_cast<float>(hostStopClock[0] - hostStartClock[0]));
    printf("Max clock rate: %0.2f GHz\n" , props.clockRate * 1e-6f);
    printf("SM counts: %d\n", props.multiProcessorCount);
    printf("NVIDIA Geforce RTX 2080 Ti GPU peak FLOPS is %f (TFLOPS)\n",
           flops * props.clockRate * 1e-9f * props.multiProcessorCount);

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
