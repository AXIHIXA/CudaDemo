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


/// LayerNorm.
/// https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
void cpuLayerNorm(const float * __restrict__ src,
                  float * __restrict__ dst,
                  int nx,
                  int ny,
                  float eps = 1e-5f,
                  float gamma = 1.0f,
                  float beta = 0.0f)
{
    for (int y = 0; y < ny; ++y)
    {
        float rowMean = 0.0f;

        for (int x = 0; x < nx; ++x)
        {
            rowMean += src[y * nx + x] / static_cast<float>(nx);
        }

        float rowVar = 0.0f;

        for (int x = 0; x < nx; ++x)
        {
            rowVar += (src[y * nx + x] - rowMean) * (src[y * nx + x] - rowMean) / static_cast<float>(nx);
        }

        float rowStd = std::sqrt(rowVar + eps);

        for (int x = 0; x < nx; ++x)
        {
            dst[y * nx + x] = (src[y * nx + x] - rowMean) / rowStd * gamma + beta;
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
    __device__ __forceinline__ T operator()(const T & a, const T & b)
    {
        return a + b;
    }
};


/// Bufferfly warp reduction.
/// See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#reduction-across-a-warp
template <template <typename> class ReductionOp, typename T, int kWarpThreads = 32>
__inline__ __device__ T warpReduce(T val)
{
    #pragma unroll
    for (int mask = kWarpThreads >> 1; 0 < mask; mask >>= 1)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask, kWarpThreads));
    }

    return val;
}


// Used for vectorized stores and loads.
template <typename T, int kSize>
struct alignas(sizeof(T) * kSize) Vec
{
    T data[kSize];
};


/// Each warp handles a consecutive of kThreadSpanY rows in src.
/// Each row in thread block constitutes a warp.
/// Grid is 1D, spans in y direction.
template <int kBlockDimX = 32, int kBlockDimY, int kThreadSpanX, int kThreadSpanY, int kPackSize, int kWarpThreads = 32>
__global__ void layerNorm(const float * __restrict__ src,
                          float * __restrict__ dst,
                          int nx,
                          int ny,
                          float eps = 1e-5f,
                          float gamma = 1.0f,
                          float beta = 0.0f)
{
    // Alignment requirements for vectorized loads/stores.
    // Each warp must be long enough to cover a row.
    // Grid must be 1D and span by the y dimension.
    if (kThreadSpanX * kWarpThreads < nx || nx % kPackSize != 0 || gridDim.x != 1 || gridDim.z != 1)
    {
        __trap();
    }

    // Each warp handles a complete row of input.
    // Each warp must consist of a complete row of threads in the thread block.
    static_assert(kWarpThreads == 32 && kBlockDimX == kWarpThreads);

    // Number of packs (vectorized load granularity on x dimension) should be integer.
    static_assert(kThreadSpanX % kPackSize == 0);

    using Vec = Vec<float, kPackSize>;
    constexpr int kPacks = kThreadSpanX / kPackSize;

    const int laneIdx = threadIdx.x;
    const int globalWarpIdx = blockIdx.y * kBlockDimY + threadIdx.y;
    const int globalWarps = gridDim.y * kBlockDimY;

    float buf[kThreadSpanY][kThreadSpanX];

    // Grid translation
    for (int baseY = globalWarpIdx * kThreadSpanY; baseY < ny; baseY += globalWarps * kThreadSpanY)
    {
        // Input, and calculate thread sum.
        float threadSum[kThreadSpanY];

        // Not all threads have kThreadSpanX elements.
        // Used when calculating row std dev.
        int threadElements[kThreadSpanY];

        #pragma unroll
        for (int rowIdx = 0; rowIdx < kThreadSpanY; ++rowIdx)
        {
            const int gy = baseY + rowIdx;
            threadSum[rowIdx] = 0;
            threadElements[rowIdx] = 0;

            if (gy < ny)
            {
                float * rowBuf = buf[rowIdx];

                #pragma unroll
                for (int packIdx = 0; packIdx < kPacks; ++packIdx)
                {
                    // p0 p0 p0 p1 p1 p1 p2 p2 p2 ...
                    const int gx = packIdx * kPackSize * kWarpThreads + laneIdx * kPackSize;
                    const int packOffset = packIdx * kPackSize;

                    if (gx < nx)
                    {
                        *reinterpret_cast<Vec *>(rowBuf + packOffset) =
                                *reinterpret_cast<const Vec *>(src + gy * nx + gx);

                        #pragma unroll
                        for (int pi = 0; pi < kPackSize; ++pi)
                        {
                            threadSum[rowIdx] += rowBuf[packOffset + pi];
                        }

                        threadElements[rowIdx] += kPackSize;
                    }
                    else
                    {
                        #pragma unroll
                        for (int pi = 0; pi < kPackSize; ++pi)
                        {
                            rowBuf[packOffset + pi] = 0;
                        }
                    }
                }
            }
        }

        // Reduce row mean.
        float rowMean[kThreadSpanY];

        #pragma unroll
        for (int rowIdx = 0; rowIdx < kThreadSpanY; ++rowIdx)
        {
            rowMean[rowIdx] = warpReduce<Sum>(threadSum[rowIdx]) / static_cast<float>(nx);
        }

        // In-place substract row mean, and reduce thread-level numerator for row variance (store into threadSum).
        #pragma unroll
        for (int rowIdx = 0; rowIdx < kThreadSpanY; ++rowIdx)
        {
            float * rowBuf = buf[rowIdx];
            threadSum[rowIdx] = 0;

            // Less than threadElements (rather than kThreadSpanX)
            // s.t. dummies do not contribute to row variance.
            for (int xi = 0; xi < threadElements[rowIdx]; ++xi)
            {
                rowBuf[xi] = rowBuf[xi] - rowMean[rowIdx];
                threadSum[rowIdx] += rowBuf[xi] * rowBuf[xi];
            }
        }

        // Reduce row std dev.
        float rowStd[kThreadSpanY];

        #pragma unroll
        for (int rowIdx = 0; rowIdx < kThreadSpanY; ++rowIdx)
        {
            float rowVar = warpReduce<Sum>(threadSum[rowIdx]) / static_cast<float>(nx);
            rowStd[rowIdx] = sqrt(rowVar + eps);
        }

        // Write-back.
        #pragma unroll
        for (int rowIdx = 0; rowIdx < kThreadSpanY; ++rowIdx)
        {
            const int gy = baseY + rowIdx;

            if (gy < ny)
            {
                float * rowBuf = buf[rowIdx];

                #pragma unroll
                for (int xi = 0; xi < kThreadSpanX; ++xi)
                {
                    rowBuf[xi] = rowBuf[xi] / rowStd[rowIdx] * gamma + beta;
                }

                #pragma unroll
                for (int packIdx = 0; packIdx < kPacks; ++packIdx)
                {
                    // p0 p0 p0 p1 p1 p1 p2 p2 p2 ...
                    const int gx = packIdx * kPackSize * kWarpThreads + laneIdx * kPackSize;
                    const int packOffset = packIdx * kPackSize;

                    if (gx < nx)
                    {
                        *reinterpret_cast<Vec *>(dst + gy * nx + gx) = *reinterpret_cast<Vec *>(rowBuf + packOffset);
                    }
                }
            }
        }
    }
}


template <typename T>
struct Equal
{
    __host__ __device__
    inline bool operator()(const T & a, const T & b) = delete;
};


template <>
struct Equal<float>
{
    __host__ __device__
    inline bool operator()(float a, float b)
    {
        return abs(a - b) < kAbsTol + kRelTol * abs(b);
    }

    static constexpr float kAbsTol = 1e-4f;
    static constexpr float kRelTol = 1e-4f;
};


template <bool kDebugOutput = true>
void checkResult(const float * __restrict__ res,
                 const float * __restrict__ gt,
                 int nx,
                 int ny)
{
    static Equal<float> equal;

    bool correct = true;

    for (int i = 0; i < nx * ny; ++i)
    {
        if (!equal(res[i], gt[i]))
        {
            correct = false;
            break;
        }
    }

    printf("result is %s\n", correct ? "correct." : "WRONG!!!");

    if constexpr (kDebugOutput)
    {
        if (correct)
        {
            return;
        }

        printf("Last three rows in res:\n");

        for (int y = 0; y < 2; ++y)
        {
            for (int x = 0; x < nx; ++x)
            {
                printf("%11.6f ", res[y * nx + x]);
            }

            printf("\n");
        }

        printf("\n\nLast three rows in gt:\n");

        for (int y = 0; y < 2; ++y)
        {
            for (int x = 0; x < nx; ++x)
            {
                printf("%11.6f ", gt[y * nx + x]);
            }
            printf("\n");
        }

        printf("\n");
    }
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
        // std::uniform_real_distribution<float> d(1, 4);
        auto g = [&e, &d]() -> float { return d(e); };
        std::generate(hostSrc.begin(), hostSrc.end(), g);
    }

    thrust::host_vector<float> gt(ny * nx, 1.0f);
    cpuLayerNorm(hostSrc.data(), gt.data(), nx, ny);

    thrust::device_vector<float> devSrc = hostSrc;
    thrust::device_vector<float> devDst(ny * nx);

    // CUDA resources that require manual destruction.
    float ms;
    cudaEvent_t ss, ee;
    CUDA_CHECK(cudaEventCreate(&ss));
    CUDA_CHECK(cudaEventCreate(&ee));

    constexpr int kPackSize = 4;
    constexpr int kWarpThreads = 32;

    constexpr dim3 kBlock(32, 8);
    constexpr int kThreadSpanX = 32;
    constexpr int kThreadSpanY = 4;
    constexpr int kBlockSpanX = kThreadSpanX * kBlock.x;
    constexpr int kBlockSpanY = kThreadSpanY * kBlock.y;

    dim3 grid((nx + kBlockSpanX - 1) / kBlockSpanX, (ny + kBlockSpanY - 1) / kBlockSpanY);

    // Test
    if constexpr (kTestSoftmax)
    {
        if constexpr (1 < kDup)
        {
            layerNorm<kBlock.x, kBlock.y, kThreadSpanX, kThreadSpanY, kPackSize, kWarpThreads><<<grid, kBlock>>>(
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
            layerNorm<kBlock.x, kBlock.y, kThreadSpanX, kThreadSpanY, kPackSize, kWarpThreads><<<grid, kBlock>>>(
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

        std::printf("layerNorm: ");
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
