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


template <typename T>
struct Max
{
    __device__ __forceinline__ constexpr bool operator()(const T & a, const T & b)
    {
        return (a < b) ? b : a;
    }
};


template <typename T>
struct Sum
{
    __device__ __forceinline__ bool operator()(const T & a, const T & b)
    {
        return a + b;
    }
};


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


// Used for vectorized stores and loads.
template <typename T, int kSize>
struct alignas(sizeof(T) * kSize) Vec
{
    T val[kSize];
};


// Vectorized load, from src's y-th row at column x, to dst.
template <int kPackSize>
__device__ void vecLoad(const float * src, int y, int nx, int x, float * dst)
{
    using Vec = Vec<float, kPackSize>;
    const int offset = (y * nx + x) / kPackSize;
    *reinterpret_cast<Vec *>(dst) = *(reinterpret_cast<Vec *>(const_cast<float *>(src)) + offset);
}


// Vectorized store, from src, to dst's y-th row at column x.
template <int kPackSize>
__device__ void vecStore(const float * src, float * dst, int y, int nx, int x)
{
    using Vec = Vec<float, kPackSize>;
    const int offset = (y * nx + x) / kPackSize;
    *(reinterpret_cast<Vec *>(dst) + offset) = *reinterpret_cast<Vec *>(src);
}


template <int kBlockDimX, int kBlockDimY, int kBlockSpanX, int kBlockSpanY, int kPackSize, int kWarpThreads = 32>
__global__ void softmax(const float * __restrict__ src,
                        float * __restrict__ dst,
                        int nx,
                        int ny)
{
    constexpr int kThreadSpanX = kBlockSpanX / kBlockDimX;
    constexpr int kThreadSpanY = kBlockSpanY / kBlockDimY;
    constexpr int kNumPacks = kThreadSpanY / kPackSize;

    // Each warp handles a complete line of input.
    assert(nx <= kThreadSpanX * kWarpThreads);
    static_assert(kBlockDimX == kWarpThreads && kWarpThreads == 32);

    // Number of packs (vectorized load granularity on x dimension) should be integer.
    static_assert(kThreadSpanX % kPackSize == 0);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int laneIdx = threadIdx.x;
    const int warpIdx = blockIdx.y * blockDim.y + threadIdx.y;  // Each row constitutes a warp.
    const int globalWarps = gridDim.y * blockDim.y;  // Total number of warps in the whole grid.
    const int yStep = globalWarps * kThreadSpanY;

    float buf[kThreadSpanY][kThreadSpanX];

    // Region of this block.
    for (int y = warpIdx * kThreadSpanY; y < ny; y += yStep)
    {
        float threadMax[kThreadSpanY];

        // Rows by this thread.
        for (int yy = 0; yy < kThreadSpanY; ++yy)
        {
            threadMax[yy] = -INFINITY;
            float * rowBuf = buf[yy];

            // One row of this thread.
            for (int packIdx = 0; packIdx < kNumPacks; ++packIdx)
            {
                const int packOffset = packIdx;
                const int x = (packIdx * kWarpThreads + laneIdx);

                if (x < nx)
                {
                    vecLoad<kPackSize>(src, y + yy, nx, x, rowBuf + packOffset);

                    for (int i = 0; i < kPackSize; ++i)
                    {
                        threadMax[yy] = max(threadMax[yy], rowBuf[packOffset + i]);
                    }
                }
                else
                {
                    for (int i = 0; i < kPackSize; ++i)
                    {
                        rowBuf[packOffset + i] = -INFINITY;
                    }
                }
            }
        }

        float warpMax[kThreadSpanY];


        for (int yy = 0; yy < kThreadSpanY; ++yy)
        {
            warpMax[yy] = warpReduce<Max, float, kWarpThreads>(threadMax[yy]);
        }

        float threadSum[kThreadSpanY];

        for (int row_id = 0; row_id < kThreadSpanY; ++row_id)
        {
            threadSum[row_id] = 0;
            float * row_buf = buf[row_id];
            // 当前线程拥有的row_buf值的总和，softmax分母的partial value
            for (int i = 0; i < cols_per_thread; ++i)
            {
                row_buf[i] = Exp(row_buf[i] - warpMax[row_id]);
                threadSum[row_id] += row_buf[i];
            }
        }
        float warp_sum[kThreadSpanY];
        // softmax分母的final value
        for (int row_id = 0; row_id < kThreadSpanY; ++row_id)
        {
            warp_sum[row_id] = WarpReduce<SumOp, float, kW>(threadSum[row_id]);
        }

        for (int row_id = 0; row_id < kThreadSpanY; ++row_id)
        {
            float * row_buf = buf[row_id];
            // 分子除分母得到sfotmax最终结果
            for (int i = 0; i < cols_per_thread; ++i)
            {
                row_buf[i] = Div(row_buf[i], warp_sum[row_id]);
            }
            // 哪里来回哪里去，把最终结果写回显存
            for (int i = 0; i < kThreadSpanX; ++i)
            {
                const int col = (i * kW + lane_id) * pack_size;
                if (col < cols)
                {
                    vecStore<pack_size>(row_buf + i * pack_size, dst, y + row_id, cols, col);
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

    constexpr int kPackSize = 1;
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
            softmax<kBlock.x, kBlock.y, kBlockSpanX, kBlockSpanY, kPackSize, kWarpThreads><<<grid, kBlock>>>(
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
            softmax<kBlock.x, kBlock.y, kBlockSpanX, kBlockSpanY, kPackSize, kWarpThreads><<<grid, kBlock>>>(
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
