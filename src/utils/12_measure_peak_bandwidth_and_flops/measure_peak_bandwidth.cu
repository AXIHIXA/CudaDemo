#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


template <int kMemoryOffset, int kPackSize = 4>
__global__ void testGmemBandwidth(float * __restrict__ A,
                                  float * __restrict__ B,
                                  float * __restrict__ C)
{
	static_assert(kPackSize == 4, "Only test with float4 s!");

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int gx = gtid; gx < kMemoryOffset / kPackSize; gx += blockDim.x * gridDim.x)
    {
		float4 a = reinterpret_cast<float4 *>(A)[gx];
		float4 b = reinterpret_cast<float4 *>(B)[gx];

        // MUST perform writes, otherwise the compiler will optimize out the loads!
		float4 c;
		c.x = a.x + b.x;
		c.y = a.y + b.y;
		c.z = a.z + b.z;
		c.w = a.w + b.w;
		reinterpret_cast<float4 *>(C)[gx] = c;
	}
}


int main(int argc, char * argv[])
{
    constexpr int kArraySize = 100000000;
    constexpr int kMemoryOffset = 10000000;
    constexpr int kDups = 10;

    constexpr bool kTestGmemBandwidth = true;
    constexpr bool kTestL2CacheBandwidth = true;

    thrust::device_vector<float> devA(kArraySize);
    thrust::device_vector<float> devB(kArraySize);
    thrust::device_vector<float> devC(kArraySize);

    float ms;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    if (kTestGmemBandwidth)
    {
        // Warmup.
        testGmemBandwidth<kDups><<<kMemoryOffset / 256 / 4, 256>>>(
                thrust::raw_pointer_cast(devA.data()),
                thrust::raw_pointer_cast(devB.data()),
                thrust::raw_pointer_cast(devC.data())
        );
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < kDups; ++i)
        {
            testGmemBandwidth<kMemoryOffset><<<kMemoryOffset / 256 / 4, 256>>>(
                    thrust::raw_pointer_cast(devA.data()),
                    thrust::raw_pointer_cast(devB.data()),
                    thrust::raw_pointer_cast(devC.data())
            );
        }

        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        thrust::host_vector<float> hostC = devC;

        if (std::ofstream fout {"/dev/null"})
        {
            fout.write(reinterpret_cast<char *>(hostC.data()),
                       hostC.size() * sizeof(float));
        }
        else
        {
            throw std::runtime_error("failed to open /dev/null");
        }

        printf("GMEM Bandwidth = %f (GB/s)\n", 3.0f * kArraySize * 4.0f / ms * 1e-6f);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

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
