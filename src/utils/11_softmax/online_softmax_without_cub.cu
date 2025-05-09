#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
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
struct alignas(sizeof(T) * 2) Mdf
{
    T m;  // max
    T d;  // exp sum
};

struct MdfOp
{
    template <typename T>
    __device__ __forceinline__ Mdf<T> operator()(const Mdf<T> & a, const Mdf<T> & b) const
    {
        Mdf<T> ans;
        ans.m = max(a.m, b.m);
        ans.d = a.d * exp(a.m - ans.m) + b.d * exp(b.m - ans.m);
        return ans;
    }
};


template <typename T, int kWarpSize = 32>
__device__ __forceinline__ Mdf<T> warpReduce(Mdf<T> data)
{
    #pragma unroll
    for (int laneMask = kWarpSize >> 1; 0 != laneMask; laneMask >>= 1)
    {
        // CUDA C++ Programming Guide
        // 7.22. Warp Shuffle Functions
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
        // T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
        Mdf<T> other;
        other.m = __shfl_xor_sync(0xffffffffu, data.m, laneMask, kWarpSize);
        other.d = __shfl_xor_sync(0xffffffffu, data.d, laneMask, kWarpSize);
        Mdf<T> ans;
        ans.m = max(data.m, other.m);
        ans.d = data.d * exp(data.m - ans.m) + other.d * exp(other.m - ans.m);
        data = ans;
    }

    return data;
}


template <int kBlockDimX, typename T, typename Op, int kWarpSize = 32>
__device__ void blockReduce(Mdf<T> & data, Op op)
{
    static_assert(0 == kBlockDimX % kWarpSize);
    constexpr int nWarps = kBlockDimX / kWarpSize;
    __shared__ Mdf<T> temp[nWarps];

    const int linear_tid = threadIdx.x;
    const int lane_id = linear_tid % kWarpSize;
    const int warp_id = linear_tid / kWarpSize;

    data = warpReduce(data);

    if (0 == lane_id)
    {
        temp[warp_id] = data;
    }

    __syncthreads();

    if (0 == linear_tid)
    {
        data = temp[0];

        #pragma unroll
        for (int i = 1; i < nWarps; ++i)
        {
            data = op(data, temp[i]);
        }
    }
}


// Online softmax, each block handles a row.
template <int kBlockDimX, typename T>
__global__ void softmax(
        const T * __restrict__ src,
        const int nCols,
        T * __restrict__ dst
        )
{
    Mdf<T> tmp;
    Mdf<T> mdf = { -INFINITY, 0 };
    __shared__ Mdf<T> rowMdf;

//    using BlockReduce = cub::BlockReduce<Mdf<T>, kBlockDimX>;
//    __shared__ typename BlockReduce::TempStorage tempStorage;

    // Block shifts across the whole row.
    for (int i = threadIdx.x; i < nCols; i += kBlockDimX)
    {
        tmp.m = src[blockIdx.x * nCols + i];
        tmp.d = 1;
        mdf = MdfOp()(mdf, tmp);
    }

//    mdf = BlockReduce(tempStorage).Reduce(mdf, MdfOp());
    blockReduce<kBlockDimX>(mdf, MdfOp());

    if (0 == threadIdx.x)
    {
        rowMdf = mdf;
    }

    __syncthreads();

    for (int i = threadIdx.x; i < nCols; i += kBlockDimX)
    {
        dst[blockIdx.x * nCols + i] = exp(src[blockIdx.x * nCols + i] - rowMdf.m) / rowMdf.d;
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

    int nx = 65536;
    int ny = 1024;
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

    constexpr dim3 kBlock(256);
    dim3 grid(ny);

    // Test
    if constexpr (kTestSoftmax)
    {
        if constexpr (1 < kDup)
        {
            softmax<kBlock.x><<<grid, kBlock>>>(
                    thrust::raw_pointer_cast(devSrc.data()),
                    nx,
                    thrust::raw_pointer_cast(devDst.data())
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        CUDA_CHECK(cudaEventRecord(ss));

        for (int dup = 0; dup < kDup; ++dup)
        {
            softmax<kBlock.x><<<grid, kBlock>>>(
                    thrust::raw_pointer_cast(devSrc.data()),
                    nx,
                    thrust::raw_pointer_cast(devDst.data())
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
