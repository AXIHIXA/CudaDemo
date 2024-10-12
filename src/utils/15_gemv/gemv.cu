#include <algorithm>
#include <cassert>
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
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


/// GEMV kernel for fp32 with cuBLAS, calculating y == alpha * (A @ x) + beta * y.
/// Used for ground truth calculation.
/// cuBLAS matrices are in COLUMN-MAJOR, A should be transposed.
/// \param[in] a      shape=(dm, dn)
/// \param[in] x      shape=(dn, 1)
/// \param[in/out] y  shape=(dn, 1)
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


/// GEVM kernel for fp32 with cuBLAS, calculating y == alpha * (x @ A) + beta * y
/// Used for ground truth calculation.
/// cuBLAS matrices are in COLUMN-MAJOR, A should be transposed.
/// \param[in] x      shape=(1, dm)
/// \param[in] a      shape=(dm, dn)
/// \param[in/out] y  shape=(1, dn)
void gevmCublas(const float * __restrict__ x,
                const float * __restrict__ a,
                float * __restrict__ y,
                int dm,
                int dn,
                float alpha,
                float beta,
                cublasHandle_t handle)
{
    CUBLAS_CHECK(
        cublasSgemv(
            handle, CUBLAS_OP_N,
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


/// alpha * A @ X + beta * Y.
/// Each 1D block computes one element in output.
template <int kBlockDimX, int kBlockDimY>
__global__ void gemvNaive(
        const float * __restrict__ a,
        const float * __restrict__ x,
        float * __restrict__ y,
        int dm,
        int dn,
        float alpha,
        float beta)
{
    // Alignment requirements for float4 vectorized loads/stores.
    // Note that this works only for Debug builds...
    assert(dm % 4 == 0);

    static_assert(kBlockDimY == 1);
    constexpr int kPackSize = 4;
    constexpr int kBlockSpanX = kPackSize * kBlockDimX;

    float4 threadData;

    // Grid Translation to cover all rows.
    for (int gy = blockIdx.y; gy < dm; gy += gridDim.x)
    {
        threadData.x = 0;
        threadData.y = 0;
        threadData.z = 0;
        threadData.w = 0;

        // Block Translation to cover all columns.
        for (int baseX = 0; baseX < dn; baseX += kBlockSpanX)
        {
            const int gx = baseX + threadIdx.x * kPackSize;

            if (gx < dn)
            {
                float4 a4 = *reinterpret_cast<const float4 *>(a + gy * dn + gx);
                float4 x4 = *reinterpret_cast<const float4 *>(x + gx);
                threadData.x += a4.x * x4.x;
                threadData.y += a4.y * x4.y;
                threadData.z += a4.z * x4.z;
                threadData.w += a4.w * x4.w;
            }
        }

        threadData.x = blockReduce<kBlockDimX, kBlockDimY>(threadData.x);
        threadData.y = blockReduce<kBlockDimX, kBlockDimY>(threadData.y);
        threadData.z = blockReduce<kBlockDimX, kBlockDimY>(threadData.z);
        threadData.w = blockReduce<kBlockDimX, kBlockDimY>(threadData.w);

        if (threadIdx.x == 0)
        {
            y[gy] = alpha * (threadData.x + threadData.y + threadData.z + threadData.w) + beta * y[gy];
        }
    }
}


__device__ __forceinline__
float4 fma(const float4 & a, float b, const float4 & c)
{
    float4 ret;
    ret.x = a.x * b + c.x;
    ret.y = a.y * b + c.y;
    ret.z = a.z * b + c.z;
    ret.w = a.w * b + c.w;
    return ret;
}


__device__ __forceinline__
float4 add(const float4 & a, const float4 & b)
{
    float4 ret;
    ret.x = a.x + b.x;
    ret.y = a.y + b.y;
    ret.z = a.z + b.z;
    ret.w = a.w + b.w;
    return ret;
}


/// GEVM
/// X [1, dn] @ A [dn, dm] --> Y [1, dm]
/// X: Q @ K.T; A: V
/// Each 1D thread block accumulates several rows in A (each row of shape [1, dm]).
/// Each thread handles a vectorized pack in one row,
/// and all threads will span multiple rows (strided w.r.t. block).
/// All elements in one row in A multiply with the same element in X.
template <int kBlockDimX, int kThreadsPerValue, typename Vec = float4>
__global__ void gevm(
        const float * __restrict__ x,
        const float * __restrict__ a,
        float * __restrict__ y,
        int dn,
        int dm)
{
    static_assert(std::is_same_v<Vec, float4>);

    constexpr int kVecSize = sizeof(Vec) / sizeof(float);
    constexpr int kBlockSpanY = kBlockDimX / kThreadsPerValue;

    extern __shared__ unsigned char smem[];
    auto rowBuf = reinterpret_cast<float *>(smem);

    const int tid = threadIdx.x;

    // This thread: (x, y) coordinate to load from matrix A.
    const int ay = tid / kThreadsPerValue;
    const int ax = (tid % kThreadsPerValue) * kVecSize;

    Vec out = {};

    // Gird translation.
    // Reduces dn rows into kBlockSpanY rows.
    for (int yy = ay; yy < dn; yy += kBlockSpanY)
    {
        Vec pack = *reinterpret_cast<const Vec *>(&a[yy * dm + ax]);
        float logits = x[yy];
        out = fma(pack, logits, out);
    }

    // Binary row-wise reduction.
    // Now we have kBlockSpanY rows spanned across this block.
    // We reduce them into one row and write back.
    for (int rowsToReduce = kBlockSpanY; 1 < rowsToReduce; rowsToReduce >>= 1)
    {
        const int mid = rowsToReduce >> 1;

        if (mid <= ay && ay < rowsToReduce)
        {
            *reinterpret_cast<Vec *>(&rowBuf[(ay - mid) * dm + ax]) = out;
        }

        __syncthreads();

        if (ay < mid)
        {
            out = add(*reinterpret_cast<Vec *>(&rowBuf[ay * dm + ax]), out);
        }

        __syncthreads();
    }

    // Write back.
    if (ay == 0)
    {
        *reinterpret_cast<Vec *>(&y[ax]) = out;
    }
}


struct alignas(16) half8
{
    half2 h1 = make_half2(0, 0);
    half2 h2 = make_half2(0, 0);
    half2 h3 = make_half2(0, 0);
    half2 h4 = make_half2(0, 0);
};


__device__ __forceinline__
half8 fma(const half8 & a, half b, const half8 & c)
{
    half8 ret;
    half2 b2 = make_half2(b, b);
    // In case __CUDA_NO_HALF2_OPERATORS__ is defined.
    ret.h1 = __hadd2(__hmul2(a.h1, b2), c.h1);
    ret.h2 = __hadd2(__hmul2(a.h2, b2), c.h2);
    ret.h3 = __hadd2(__hmul2(a.h3, b2), c.h3);
    ret.h4 = __hadd2(__hmul2(a.h4, b2), c.h4);
    return ret;
}


__device__ __forceinline__
half8 add(const half8 & a, const half8 & b)
{
    half8 ret;
    ret.h1 = __hadd2(a.h1, b.h1);
    ret.h2 = __hadd2(a.h2, b.h2);
    ret.h3 = __hadd2(a.h3, b.h3);
    ret.h4 = __hadd2(a.h4, b.h4);
    return ret;
}


///// GEVM for FP16.
///// X [1, dn] @ A [dn, dm] --> Y [1, dm]
///// X: Q @ K.T; A: V
///// Each 1D thread block accumulates several rows in A (each row of shape [1, dm]).
///// Each thread handles a vectorized pack in one row,
///// and all threads will span multiple rows (strided w.r.t. block).
///// All elements in one row in A multiply with the same element in X.
template <int kBlockDimX, int kThreadsPerValue, typename Vec = half8>
__global__ void gevm(
        const half * __restrict__ x,
        const half * __restrict__ a,
        half * __restrict__ y,
        int dn,
        int dm)
{
    static_assert(std::is_same_v<Vec, half8>);

    constexpr int kVecSize = sizeof(Vec) / sizeof(half);
    constexpr int kBlockSpanY = kBlockDimX / kThreadsPerValue;

    extern __shared__ unsigned char smem[];
    auto rowBuf = reinterpret_cast<half *>(smem);

    const int tid = threadIdx.x;

    // This thread: (x, y) coordinate to load from matrix A.
    const int ay = tid / kThreadsPerValue;
    const int ax = (tid % kThreadsPerValue) * kVecSize;

    half8 out = {};

    // Gird translation.
    // Reduces dn rows into kBlockSpanY rows.
    for (int yy = ay; yy < dn; yy += kBlockSpanY)
    {
        half8 pack = *reinterpret_cast<const half8 *>(&a[yy * dm + ax]);
        half logits = x[yy];
        out = fma(pack, logits, out);
    }

    // Binary row-wise reduction.
    // Now we have kBlockSpanY rows spanned across this block.
    // We reduce them into one row and write back.
    for (int rowsToReduce = kBlockSpanY; 1 < rowsToReduce; rowsToReduce >>= 1)
    {
        const int mid = rowsToReduce >> 1;

        if (mid <= ay && ay < rowsToReduce)
        {
            *reinterpret_cast<Vec *>(&rowBuf[(ay - mid) * dm + ax]) = out;
        }

        __syncthreads();

        if (ay < mid)
        {
            out = add(*reinterpret_cast<Vec *>(&rowBuf[ay * dm + ax]), out);
        }

        __syncthreads();
    }

    // Write back.
    if (ay == 0)
    {
        *reinterpret_cast<Vec *>(&y[ax]) = out;
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
    bool resultIsCorrect = thrust::equal(thrust::device, result.cbegin(), result.cend(), golden.cbegin(), Equal<acc_t>());

    if constexpr (kDebugOutput)
    {
        if (!resultIsCorrect)
        {
            thrust::host_vector<acc_t> a = result;
            thrust::host_vector<acc_t> b = golden;

            printf("Result:\n");
            for (int i = 0; i < dm; ++i)
            {
                printf("%6.2f ", a[i]);
            }
            printf("\n\n");

            printf("Ground truth:\n");
            for (int i = 0; i < dm; ++i)
            {
                printf("%6.2f ", b[i]);
            }
            printf("\n\n");
        }
    }

    std::printf("Result is %s\n\n", resultIsCorrect ? "correct." : "WRONG!!!");
}


void displayInputs(const thrust::host_vector<float> & h_a,
                   const thrust::host_vector<float> & h_x,
                   int m,
                   int n,
                   float alpha,
                   float beta)
{
    printf("alpha = %f, beta = %f\n", alpha, beta);

    printf("A:\n");
    for (int y = 0; y < m; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            printf("%6.0f ", h_a[y * n + x]);
        }
        printf("\n");
    }

    printf("X:\n");
    for (int y = 0; y < m; ++y)
    {
        printf("%6.0f ", h_x[y]);
    }
    printf("\n");
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

    constexpr bool kTestGemvNaive = false;
    constexpr bool kTestGevmFp32 = false;
    constexpr bool kTestGevmFp16 = true;

    // Problem setting.
    const int problemSize = 1024;
    const int m = problemSize;
    const int n = problemSize;
    float alpha = 1.0f;
    float beta = 0.0f;
    thrust::host_vector<float> h_a(m * n, 1.0f);
    thrust::host_vector<float> h_x(m, 1.0f);
    thrust::host_vector<float> h_y(std::max(m, n), 0.0f);
//    std::iota(h_a.begin(), h_a.end(), 1.0f);
//    std::iota(h_x.begin(), h_x.end(), 1.0f);

    if constexpr (kRandInput)
    {
        unsigned seed = std::random_device()();
        // std::printf("seed = %u\n", seed);
        std::default_random_engine e(seed);
        // std::normal_distribution<float> d(0.0f, 1.0f);
        std::uniform_int_distribution d(1, 20);
        auto g = [&e, &d]() -> float { return d(e); };
        alpha = g();
        beta = g();
        std::generate(h_a.begin(), h_a.end(), g);
        std::generate(h_x.begin(), h_x.end(), g);
        std::generate(h_y.begin(), h_y.end(), g);
    }

    // displayInputs(h_a, h_x, m, n, alpha, beta);

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

    // GEMV Naive.
    if (kTestGemvNaive)
    {
        // Compute GEMV ground truth with cuBLAS.
        gemvCublas(thrust::raw_pointer_cast(d_a.data()),
                   thrust::raw_pointer_cast(d_x.data()),
                   thrust::raw_pointer_cast(d_y.data()),
                   m,
                   n,
                   alpha,
                   beta,
                   handle);
        golden_y = d_y;

        // GEMV
        d_y = h_y;

        constexpr int kBlockDimX = 128;
        constexpr int kBlockDimY = 1;
        constexpr dim3 kBlock(kBlockDimX, kBlockDimY);
        dim3 grid(std::min(1, m >> 1));

        gemvNaive<kBlockDimX, kBlockDimY><<<grid, kBlock>>>(
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

    if constexpr (kTestGevmFp32)
    {
        // Compute GEMV ground truth with cuBLAS.
        gemvCublas(thrust::raw_pointer_cast(d_a.data()),
                   thrust::raw_pointer_cast(d_x.data()),
                   thrust::raw_pointer_cast(d_y.data()),
                   m,
                   n,
                   alpha,
                   beta,
                   handle);

        golden_y = d_y;

        // GEVM
        d_y = h_y;

        constexpr int kBlockDimX = 256;
        using T = float;
        using Vec = float4;
        constexpr int kThreadsPerValue = n * sizeof(T) / sizeof(Vec);
        static_assert(kBlockDimX % kThreadsPerValue == 0);
        const int smemBytes = (1 + kBlockDimX / kThreadsPerValue) * n * sizeof(T);

        gevm<kBlockDimX, kThreadsPerValue, Vec><<<1, kBlockDimX, smemBytes>>>(
                thrust::raw_pointer_cast(d_x.data()),
                thrust::raw_pointer_cast(d_a.data()),
                thrust::raw_pointer_cast(d_y.data()),
                m,
                n
        );

        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::printf("FP32 GEVM ");
        checkResult(d_y, golden_y, n);
    }

    if constexpr (kTestGevmFp16)
    {
        // GEVM
        thrust::device_vector<half> d_x(1 * m, static_cast<half>(1));
        thrust::device_vector<half> d_a(m * n, static_cast<half>(1));
        thrust::device_vector<half> d_y(1 * n, static_cast<half>(0));

        constexpr int kBlockDimX = 256;
        using T = half;
        using Vec = half8;
        constexpr int kThreadsPerValue = n * sizeof(T) / sizeof(Vec);
        static_assert(kBlockDimX % kThreadsPerValue == 0);
        const int smemBytes = (1 + kBlockDimX / kThreadsPerValue) * n * sizeof(T);

        gevm<kBlockDimX, kThreadsPerValue, Vec><<<1, kBlockDimX, smemBytes>>>(
                thrust::raw_pointer_cast(d_x.data()),
                thrust::raw_pointer_cast(d_a.data()),
                thrust::raw_pointer_cast(d_y.data()),
                m,
                n
        );

        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::host_vector<half> h_x = d_x;

        for (int i = 0; i < n; ++i)
        {
            std::printf("%6.2f ", static_cast<float>(h_x[i]));
        }

        std::printf("\n");
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


/*
Compute sanitizer to check memory access:
compute-sanitizer --tool memcheck ./cmake-build-debug/demo
*/