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


/// Softmax on innermost dimension.
void cpuSoftmax(const float * __restrict__ in, float * __restrict__ out, int nx, int ny)
{
    for (int y = 0; y < ny; ++y)
    {
        float rowSum = 0.0f;
        float rowMax = std::numeric_limits<float>::min();

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
    T val[kSize];
};


template <int kPackSize>
__device__ void vecLdSt(const float * src, float * dst)
{
    using Vec = Vec<float, kPackSize>;
    *reinterpret_cast<Vec *>(dst) = *(reinterpret_cast<const Vec *>(src));
}


/// Each warp handles a consecutive of kThreadSpanY rows in src.
/// Each row in thread block constitutes a warp.
/// Grid is 1D, spans in y direction.
template <int kBlockDimX = 32, int kBlockDimY, int kThreadSpanX, int kThreadSpanY, int kPackSize, int kWarpThreads = 32>
__global__ void softmax(const float * __restrict__ src,
                        float * __restrict__ dst,
                        int nx,
                        int ny)
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

    constexpr float kMinusInfinity = -10000000000.0f;
    constexpr int kNumPacks = kThreadSpanX / kPackSize;

    const int tid = threadIdx.y * kBlockDimX + threadIdx.x;
    const int laneIdx = tid % kWarpThreads;
    const int globalWarpIdx = blockIdx.y * kBlockDimY + threadIdx.y;

    // Register.
    float buf[kThreadSpanY][kThreadSpanX];

    // Grid translation.
    for (int baseY = globalWarpIdx * kThreadSpanY; baseY < ny; baseY += gridDim.y * kBlockDimY * kThreadSpanY)
    {
        float threadMax[kThreadSpanY];

        // Rows handled by a warp.
        // Load into register and reduce thread max.
        #pragma unroll
        for (int rowIdx = 0; rowIdx < kThreadSpanY; ++rowIdx)
        {
            const int gy = baseY + rowIdx;

            // Must be outside, other threadMax might have uninitalized values!
            float * rowBuf = buf[rowIdx];
            threadMax[rowIdx] = kMinusInfinity;

            if (gy < ny)
            {
                // Vectorized load packs in this warp (row), by this thread.
                // Each threads loads kThreadSpanX elements into its rowBuf (GMEM -> REG).
                // Loads are done by kNumPacks vectorized packs, each pack of length kPackSize.
                // Packs by the same threads are strided, so that warp GMEM loads are coalesced.
                #pragma unroll
                for (int packIdx = 0; packIdx < kNumPacks; ++packIdx)
                {
                    // p0 p0 p0  p1 p1 p1  p2 p2 p2
                    const int gx = packIdx * kPackSize * kWarpThreads + laneIdx * kPackSize;
                    const int packOffset = packIdx * kPackSize;

                    if (gx < nx)
                    {
                        vecLdSt<kPackSize>(src + gy * nx + gx, rowBuf + packOffset);

                        #pragma unroll
                        for (int pi = 0; pi < kPackSize; ++pi)
                        {
                            threadMax[rowIdx] = max(threadMax[rowIdx], rowBuf[packOffset + pi]);
                        }
                    }
                    else
                    {
                        #pragma unroll
                        for (int pi = 0; pi < kPackSize; ++pi)
                        {
                            rowBuf[packOffset + pi] = kMinusInfinity;
                        }
                    }
                }
            }
        }

        // Reduce per-warp row max.
        float rowMax[kThreadSpanY];

        #pragma unroll
        for (int rowIdx = 0; rowIdx < kThreadSpanY; ++rowIdx)
        {
            rowMax[rowIdx] = warpReduce<Max, float>(threadMax[rowIdx]);
        }

        // In-place exp(x - rowmax), accumulate thread sum.
        float threadSum[kThreadSpanY];

        #pragma unroll
        for (int rowIdx = 0; rowIdx < kThreadSpanY; ++rowIdx)
        {
            float * rowBuf = buf[rowIdx];
            threadSum[rowIdx] = 0;

            #pragma unroll
            for (int xi = 0; xi < kThreadSpanX; ++xi)
            {
                rowBuf[xi] = exp(rowBuf[xi] - rowMax[rowIdx]);
                threadSum[rowIdx] += rowBuf[xi];
            }
        }

        // Reduce per-warp row sum.
        float rowSum[kThreadSpanY];

        #pragma unroll
        for (int rowIdx = 0; rowIdx < kThreadSpanY; ++rowIdx)
        {
            rowSum[rowIdx] = warpReduce<Sum, float>(threadSum[rowIdx]);
        }

        // In-place softmax division by rowSum and write-back.
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
                    rowBuf[xi] /= rowSum[rowIdx];
                }

                #pragma unroll
                for (int packIdx = 0; packIdx < kNumPacks; ++packIdx)
                {
                    // p0 p0 p0  p1 p1 p1  p2 p2 p2
                    const int gx = packIdx * kPackSize * kWarpThreads + laneIdx * kPackSize;
                    const int packOffset = packIdx * kPackSize;

                    if (gx < nx)
                    {
                        vecLdSt<kPackSize>(rowBuf + packOffset, dst + gy * nx + gx);
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

        printf("res:\n");

        for (int y = 0; y < 2; ++y)
        {
            for (int x = 0; x < nx; ++x)
            {
                printf("%11.6f ", res[y * nx + x]);
            }

            printf("\n");
        }

        printf("\n\ngt :\n");

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
        // std::uniform_int_distribution<int> d(1, 10);
        auto g = [&e, &d]() -> float { return d(e); };
        std::generate(hostSrc.begin(), hostSrc.end(), g);
    }

    thrust::host_vector<float> gt(ny * nx, 1.0f);
    cpuSoftmax(hostSrc.data(), gt.data(), nx, ny);

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
    constexpr int kThreadSpanY = 1;
    constexpr int kBlockSpanX = kThreadSpanX * kBlock.x;
    constexpr int kBlockSpanY = kThreadSpanY * kBlock.y;

    dim3 grid((nx + kBlockSpanX - 1) / kBlockSpanX, (ny + kBlockSpanY - 1) / kBlockSpanY);

    // Test
    if constexpr (kTestSoftmax)
    {
        if constexpr (1 < kDup)
        {
            softmax<kBlock.x, kBlock.y, kThreadSpanX, kThreadSpanY, kPackSize, kWarpThreads><<<grid, kBlock>>>(
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
            softmax<kBlock.x, kBlock.y, kThreadSpanX, kThreadSpanY, kPackSize, kWarpThreads><<<grid, kBlock>>>(
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
