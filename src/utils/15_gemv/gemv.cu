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
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


/// GEMM kernel for fp32 with cuBLAS, calculating y == alpha * (A @ x) + beta.
/// Used for ground truth calculation.
/// cuBLAS matrices are in COLUMN-MAJOR, A should be transposed.
/// \param[in] A      shape=(dm, dn)
/// \param[in/out] B  shape=(dn, 1)
void gemvCublas(const float * __restrict__ a,
                const float * __restrict__ x,
                float * __restrict__ y,
                int dm,
                int dn,
                float alpha,
                float beta,
                cublasHandle_t handle)
{
    CUBLAS_CHECK(
        cublasSgemv(
            handle, CUBLAS_OP_T,
            dm, dn,
            &alpha,
            a, dn,
            x, 1,
            &beta,
            y, 1
        )
    );
}


template <typename acc_t, int kWarpThreads = 32>
__device__ acc_t warpReduce(acc_t val)
{
    static_assert(kWarpThreads == 32);

    #pragma unroll
    for (int step = (kWarpThreads >> 1); 0 < step; step >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, step, kWarpThreads);
    }

    return val;
}


template <int kBlockDimX, int kBlockDimY, int kWarpThreads = 32, typename acc_t>
__device__ acc_t blockReduce(acc_t val)
{
    static_assert(kBlockDimX % kWarpThreads == 0 && kBlockDimY == 1 && kWarpThreads == 32);
    constexpr int kWarps = kBlockDimX / kWarpThreads;
    val = warpReduce(val);

    __shared__ acc_t warpAggregate[kWarps];

    const int laneIdx = threadIdx.x % kWarpThreads;
    const int warpIdx = threadIdx.x / kWarpThreads;

    if (warpIdx < kWarps && laneIdx == 0)
    {
        warpAggregate[warpIdx] = val;
    }

    __syncthreads();

    val = 0;

    #pragma unroll
    for (int warp = 0; warp < kWarps; ++warp)
    {
        val += warpAggregate[warp];
    }

    return val;
}


/// Each 1D block computes one element in output.
template <
        int kBlockDimX, int kBlockDimY,
        int kBlockSpanX, int kBlockSpanY,
        int kPadding = 4,
        typename in_t, typename acc_t>
__global__ void gemvNaive(
        const in_t * __restrict__ a,
        const in_t * __restrict__ x,
        acc_t * __restrict__ y,
        int dm,
        int dn,
        acc_t alpha,
        acc_t beta)
{
    static_assert(std::is_same_v<in_t, float>);
    static_assert(kBlockSpanX % kBlockDimX == 0);
    static_assert(kBlockDimY == 1 && kBlockSpanY == 1);
    constexpr int kThreadSpanX = kBlockSpanX / kBlockDimX;
    constexpr int kPackSize = 1;

    acc_t threadData;

    // Grid Translation.
    for (int gy = blockIdx.y * kBlockSpanY; gy < dm; gy += gridDim.x * kBlockSpanY)
    {
        threadData = 0;

        #pragma unroll
        for (int packIdx = 0; packIdx < kThreadSpanX; ++packIdx)
        {
            // p0 p0 p0 ... p0   p1 p1 p1 ... p1   ...
            const int gx = packIdx * kPackSize * kBlockDimX + threadIdx.x * kPackSize;

            if (gx < dn)
            {
                threadData += a[gy * dn + gx] * x[gx];
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

    static constexpr float kAbsTol = 1e-3f;
    static constexpr float kRelTol = 2e-4f;
};


template <bool kDebugOutput = true, typename acc_t>
void checkResult(const thrust::device_vector<acc_t> & result,
                 const thrust::device_vector<acc_t> & golden,
                 int dm)
{
    if constexpr (kDebugOutput)
    {
        thrust::host_vector<acc_t> a = result;
        thrust::host_vector<acc_t> b = golden;

        printf("Result:\n");
        for (int i = 0; i < dm; ++i)
        {
            printf("%f ", a[i]);
            printf("\n");
        }
        printf("\n\n");

        printf("Ground truth:\n");
        for (int i = 0; i < dm; ++i)
        {
            printf("%f ", b[i]);
            printf("\n");
        }
        printf("\n\n");
    }

    bool resultIsCorrect = thrust::equal(thrust::device, result.cbegin(), result.cend(), golden.cbegin(), Equal<acc_t>());
    std::printf("Result is %s\n\n", resultIsCorrect ? "correct." : "WRONG!!!");
}


int main(int argc, char * argv[])
{
    /// Switches for debugging output correctness.
    /// \param kDup        Set to 1 to debug output (kernel only launched once) and results will be checked.
    ///                    Set to values greater than 1 to profile.
    ///                    In the latter case, results will NOT be checked because it's in-place GEMM.
    ///                    We do not dispatch by build type because we have -G flag for Debug builds
    ///                    (that's for debugging runtime errors).
    /// \param kRandInput  Whether we random input matrices.
    ///                    Enable when checking correctness or profiling.
    ///                    Disable when debugging output.
    constexpr int kDup = 1;
    constexpr bool kRandInput = false;

    constexpr bool kTestGemvNaive = true;
    constexpr bool kTestGemvSmem = true;

    // Problem setting.
    // Tested on NVIDIA Geforce RTX 2080 Ti (kDup=100, kRandInput=true),
    // gemmSmemPad reaches 80% cublasSgemm performance on m=n=k=2048,
    //                     90% cublasSgemm performance on m=n=k=4096.
    // It shows that CUBLAS_DEFAULT_MATH defaults to CUBLAS_TF32_TENSOR_OP_MATH.
    int problemSize = 4;
    int m = problemSize;
    int n = problemSize;
    float alpha = 1.0f;
    float beta = 0.0f;
    thrust::host_vector<float> h_a(m * n, 1.0f);
    thrust::host_vector<float> h_x(m, 1.0f);
    thrust::host_vector<float> h_y(m, 0.0f);
    std::iota(h_a.begin(), h_a.end(), 1.0f);
    std::iota(h_x.begin(), h_x.end(), 1.0f);

    if constexpr (kRandInput)
    {
        unsigned seed = std::random_device()();
        std::default_random_engine e(seed);
        // std::normal_distribution<float> d(0.0f, 1.0f);
        std::uniform_int_distribution d(1, 20);
        auto g = [&e, &d]() -> float { return d(e); };
        std::generate(h_a.begin(), h_a.end(), g);
        std::generate(h_x.begin(), h_x.end(), g);
        std::generate(h_y.begin(), h_y.end(), g);
    }

    thrust::device_vector<float> golden_y(n);
    thrust::device_vector<float> d_a = h_a;
    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;

    // CUDA resources that require manual destruction.
    float ms;
    cudaEvent_t ss, ee;
    CUDA_CHECK(cudaEventCreate(&ss));
    CUDA_CHECK(cudaEventCreate(&ee));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Testing says that these two modes are the same.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    // CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    // Compute ground truth with cuBLAS.
    gemvCublas(thrust::raw_pointer_cast(d_a.data()),
               thrust::raw_pointer_cast(d_x.data()),
               thrust::raw_pointer_cast(d_y.data()),
               m,
               n,
               alpha,
               beta,
               handle);
    golden_y = d_y;

    constexpr int kBlockDimX = 128;
    constexpr int kBlockDimY = 1;
    constexpr int kBlockSpanX = 128;
    constexpr int kBlockSpanY = 1;
    constexpr dim3 kBlock(kBlockDimX, kBlockDimY);
    dim3 grid(std::min(1, m >> 1));

    // GEMV Naive.
    if (kTestGemvNaive)
    {
        gemvNaive<128, 1, kBlockSpanX, kBlockSpanY><<<grid, kBlock>>>(
               thrust::raw_pointer_cast(d_a.data()),
               thrust::raw_pointer_cast(d_x.data()),
               thrust::raw_pointer_cast(d_y.data()),
               m,
               n,
               alpha,
               beta
        );
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
        checkResult(d_y, golden_y, m);
    }

    // Free cuda resources.
    CUDA_CHECK(cudaEventDestroy(ss));
    CUDA_CHECK(cudaEventDestroy(ee));
    CUBLAS_CHECK(cublasDestroy_v2(handle));

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
