#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


/// Softmax on innermost dimension.
void cpuSoftmax(const float * __restrict__ in, float * __restrict__ out, int nx, int ny)
{
    for (int y = 0; y < ny; ++y)
    {
        float rowSum = 0.0f;
        float rowMax = 0.0f;

        for (int x = 0; x < nx; ++x)
        {
            rowMax = std::max(in[y * nx + x], rowMax);
        }

        for (int x = 0; x < nx; ++x)
        {
            rowSum += std::exp(in[y * nx + x] - rowMax);
        }

        for (int x = 0; x < nx; ++x)
        {
            out[y * nx + x] = std::exp(in[y * nx + x] - rowMax) / rowSum;
        }
    }
}


template <template <typename> class ReductionOp, typename T, int kWarpThreads = 32>
__inline__ __device__ T warpReduce(T val)
{
    #pragma unroll
    for (int mask = kWarpThreads >> 1; 0 < mask; mask >>= 1)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }

    return val;
}


template <int kBlockDimX, int kBlockDimY, int kBlockSpanX, int kBlockSpanY, int kWarpThreads = 32>
__global__ void softmax(const float * __restrict__ src,
                        float * __restrict__ dst,
                        int nx,
                        int ny)
{
    constexpr dim3 kThreadSpan(kBlockSpanX / kBlockDimX, kBlockSpanY / kBlockDimY);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int laneIdx = tid % kWarpThreads;
    const int warpIdx = tid / kWarpThreads;

    const int bx = blockIdx.x * kBlockSpanX;
    const int by = blockIdx.y * kBlockSpanY;

    float buf[kThreadSpan.y][kThreadSpan.x] = {};

    float threadMax = -INFINITY;

    #pragma unroll
    for (int yi = 0; yi < kThreadSpan.y; ++yi)
    {
        int y = by + kThreadSpan.y * yi;

        if (y < ny)
        {
            #pragma unroll
            for (int xi = 0; xi < kThreadSpan.x; ++xi)
            {
                int x = bx + kThreadSpan.x * xi;

                if (x < nx)
                {
                    threadMax = max(threadMax, src[y * nx + x]);
                }
            }
        }
    }


}


template <bool kDebugOutput = false>
void checkResult(const float * __restrict__ res,
                 const float * __restrict__ gt,
                 int nx,
                 int ny)
{
    bool correct = true;

    for (int i = 0; i < nx * ny; ++i)
    {
        if (res[i] != gt[i])
        {
            correct = false;
            break;
        }
    }

    std::printf("result is %s\n", correct ? "correct." : "WRONG!!!");

    #if kDebugOutput
    printf("res:\n");
    for (int y = 0; y < ny; ++y)
    {
        for (int x = 0;x < nx; ++x)
        {
            printf("%5.4f ", res[i]);
        }
        printf("\n");
    }
    printf("\n\ngt :\n");
    for (int y = 0; y < ny; ++y)
    {
        for (int x = 0; x < nx; ++x)
        {
            printf("%5.4f ", gt[i]);
        }
        printf("\n");
    }
    printf("\n");
    #endif  // kDebugOutput
}


int main(int argc, char * argv[])
{
    /// Switches for debugging output correctness.
    /// \param kDup        Set to 1 to debug output (kernel only launched once) and results will be checked.
    ///                    Set to values greater than 1 to profile.
    ///                    In the latter case, results will NOT be checked because it's in-place GEMM.
    ///                    We do not dispatch by build type because we have -G flag for Debug builds
    ///                    (that's for debugging runtime errors).
    /// \param kRandInput  Whether we random input.
    ///                    Enable when checking correctness or profiling.
    ///                    Disable when debugging output.
    constexpr int kDup = 1;
    constexpr bool kRandInput = true;

    constexpr bool kTestSoftmax = true;

    int nx = 1024;
    int ny = 1000;
    thrust::host_vector<float> hostSrc(ny * nx, 1.0f);
    thrust::host_vector<float> hostDst;

    if constexpr (kRandInput)
    {
        unsigned seed = std::random_device()();
        std::default_random_engine e(seed);
        std::normal_distribution<float> d(4.0f, 1.0f);
        auto g = [&e, &d]() -> float { return d(e); };
        std::generate(hostSrc.begin(), hostSrc.end(), g);
    }

    thrust::host_vector<float> gt(ny * nx);
    cpuSoftmax(hostSrc.data(), gt.data(), nx, ny);

    thrust::device_vector<float> devSrc = hostSrc;
    thrust::device_vector<float> devDst(ny * nx);

    // CUDA resources that require manual destruction.
    float ms;
    cudaEvent_t ss, ee;
    CUDA_CHECK(cudaEventCreate(&ss));
    CUDA_CHECK(cudaEventCreate(&ee));

    constexpr int kWarpThreads = 32;
    constexpr dim3 kBlock(32, 8);
    constexpr int kBlockSpanX = kBlock.x * 32;
    constexpr int kBlockSpanY = kBlock.y;
    dim3 grid((nx + kBlockSpanX - 1) / kBlockSpanX, (ny + kBlockSpanY - 1) / kBlockSpanY);

    // Test
    if constexpr (kTestSoftmax)
    {
        if constexpr (1 < kDup)
        {
            softmax<kBlock.x, kBlock.y, kBlockSpanX, kBlockSpanY, kWarpThreads><<<grid, kBlock>>>(
                    thrust::raw_pointer_cast(devSrc.data()),
                    thrust::raw_pointer_cast(devDst.data()),
                    nx,
                    ny
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        CUDA_CHECK(cudaEventRecord(ss));

        for (int dup = 0; dup < kDup; ++dup)
        {
            softmax<kBlock.x, kBlock.y, kBlockSpanX, kBlockSpanY, kWarpThreads><<<grid, kBlock>>>(
                    thrust::raw_pointer_cast(devSrc.data()),
                    thrust::raw_pointer_cast(devDst.data()),
                    nx,
                    ny
            );
        }

        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ee));
        CUDA_CHECK(cudaEventSynchronize(ee));

        hostDst = devDst;

        std::printf("softmax: ");
        CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
        std::printf("took %f ms, ", ms / kDup);

        if constexpr (1 == kDup)
        {
            checkResult(hostDst.data(), gt.data(), nx, ny);
        }
        else
        {
            std::printf("\n");
        }
    }

    // Free cuda resources.
    CUDA_CHECK(cudaEventDestroy(ss));
    CUDA_CHECK(cudaEventDestroy(ee));

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
