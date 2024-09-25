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


/// Naive CPU GEMM kernel calculating alpha * (A @ B) + beta * C.
/// Just for demo purpose. Do not call this method. It's SUPER SLOW!
/// \param[in] A      shape=(dm, dk)
/// \param[in] B      shape=(dk, dn)
/// \param[in/out] C  shape=(dm, dn)
template <typename in_t, typename acc_t>
void gemmCpuNaive(const in_t * __restrict__ a,
                  const in_t * __restrict__ b,
                  acc_t * __restrict__ c,
                  int dm,
                  int dn,
                  int dk,
                  acc_t alpha,
                  acc_t beta)
{
    std::vector<acc_t> tmp(dm * dn, 0);

    for (int k = 0; k < dk; ++k)
    {
        for (int m = 0; m < dm; ++m)
        {
            for (int n = 0; n < dn; ++n)
            {
                tmp[m * dn + n] += a[m * dk + k] * b[k * dn + n];
            }
        }
    }

    for (int m = 0; m < dm; ++m)
    {
        for (int n = 0; n < dn; ++n)
        {
            c[m * dn + n] = alpha * tmp[m * dn + n] + beta * c[m * dn + n];
        }
    }
}


/// GEMM kernel for fp32 with cuBLAS, calculating alpha * (A @ B) + beta * C.
/// Used for ground truth calculation.
/// Since cuBLAS matrices are in COLUMN-MAJOR:
/// Note that matrix A stored in row-major EQUALS A.T stored in column major.
/// We just let cuBLAS compute B.T @ A.T.
/// the result is (AB).T in column major, which equals AB in row major!
/// And, B.T and A.T (in column major) equals B and A (in row major).
/// So just pass in B and A (in row major, not transposed) and corresponding shapes (transposed).
/// \param[in] A      shape=(dm, dk)
/// \param[in] B      shape=(dk, dn)
/// \param[in/out] C  shape=(dm, dn)
void gemmCublas(const float * __restrict__ a,
                const float * __restrict__ b,
                float * __restrict__ c,
                int dm,
                int dn,
                int dk,
                float alpha,
                float beta)
{
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dn, dm, dk,
            &alpha,
            b, dn,
            a, dk,
            &beta,
            c, dn
        )
    );

    CUBLAS_CHECK(cublasDestroy_v2(handle));
}


/// Naive GEMM kernel calculating alpha * (A @ B) + beta * C.
/// Used for baseline.
/// Each thread block has (kBlockSize, kBlockSize) threads,
/// Computing a square of (kBlockM, kBlockN) elements in output matrix.
/// Thus each thread computes a square of (kBlockM, kBlockN) / kBlockSize elements.
/// \param[in] A      shape=(dm, dk)
/// \param[in] B      shape=(dk, dn)
/// \param[in/out] C  shape=(dm, dn)
template <int kBlockSize, int kBlockM, int kBlockN, int kBlockK, typename in_t, typename acc_t>
__global__ void gemmNaive(const in_t * __restrict__ a,
                          const in_t * __restrict__ b,
                          acc_t * __restrict__ c,
                          int dm,
                          int dn,
                          int dk,
                          acc_t alpha,
                          acc_t beta)
{
    // The x-span (N) and y-span (M) of this thread.
    // This thread computes (kThreadM, kThreadN) elements in the output matrix shaped (dm, dn).
    constexpr int kThreadM = kBlockM / kBlockSize;
    constexpr int kThreadN = kBlockN / kBlockSize;

    // Top-left corner of this thread's elements in output matrix.
    int tx = blockIdx.x * kBlockN + threadIdx.x * kThreadN;  // + n, 0 <= n < kThreadN
    int ty = blockIdx.y * kBlockM + threadIdx.y * kThreadM;  // + m, 0 <= m < kThreadM

    acc_t reg[kThreadM][kThreadN] = {0};

    for (int k = 0; k < dk; ++k)
    {
        for (int m = 0; m < kThreadM; ++m)
        {
            for (int n = 0; n < kThreadN; ++n)
            {
                if (ty + m < dm && tx + n < dn)
                {
                    reg[m][n] += a[(ty + m) * dk + k] * b[k * dn + (tx + n)];
                }
            }
        }
    }

    for (int m = 0; m < kThreadM; ++m)
    {
        for (int n = 0; n < kThreadN; ++n)
        {
            if (ty + m < dm && tx + n < dn)
            {
                c[(ty + m) * dn + (tx + n)] =
                        alpha * reg[m][n] +
                        beta * c[(ty + m) * dn + (tx + n)];
            }
        }
    }
}


/// GEMM kernel calculating alpha * (A @ B) + beta * C.
/// Utilizes shared memory (padded to avoid bank conflict),
/// and vectorized loads/stores (GMEM <-> SMEM).
///
/// Each thread block has (kBlockSize, kBlockSize) threads,
/// Computing a square of (kBlockM, kBlockN) elements in output matrix.
/// Thus each thread computes a square of (kBlockM, kBlockN) / kBlockSize elements.
///
/// It's reasonable to tile-by-df-dimension,
/// as each thread block computes by kBlockM * kBlockN * dk,
/// and accesses GMEM by kBlockM * dk + kBlockN * dk,
/// the compute-memory ratio is 1 / ( 1/kBlockM + 1/kBlockN ),
/// which is irrelevant with dimension dk.
///
/// Moreover, we take the "outer product" way to accumulate elements in output.
/// Each time a thread block loads
/// (kBlockM, kBlockK) elements from A,
/// (kBlockK, kBlockN) elements from B,
/// and this sliding window slides in a sequential manner.
/// \param[in] A      shape=(dm, dk)
/// \param[in] B      shape=(dk, dn)
/// \param[in/out] C  shape=(dm, dn)
template <int kBlockSize, int kBlockM, int kBlockN, int kBlockK, int kPadding = 1>
__global__ void gemmSmem(const float * __restrict__ a,
                         const float * __restrict__ b,
                         float * __restrict__ c,
                         int dm,
                         int dn,
                         int dk,
                         float alpha,
                         float beta)
{
    // The x-span (N) and y-span (M) of this thread.
    // This thread computes (kThreadM, kThreadN) elements in the output matrix shaped (dm, dn).
    constexpr int kThreadM = kBlockM / kBlockSize;
    constexpr int kThreadN = kBlockN / kBlockSize;

    // Top-left corner of this thread's elements in output matrix.
    int baseX = blockIdx.x * kBlockN + threadIdx.x * kThreadN;  // + n, 0 <= n < kThreadN
    int baseY = blockIdx.y * kBlockM + threadIdx.y * kThreadM;  // + m, 0 <= m < kThreadM

    // Indexes.
    int baseIdx = threadIdx.y * kBlockSize + threadIdx.x;  // Index of this thread in block.
    int laneIdx = baseIdx & 31;                            // tid % 32
    int warpIdx = baseIdx >> 5;                            // tid / 32

    // Read a and b chunks into smem.
    // A chunk of a has shape (kBlockM, kBlockK) == (128, 8).
    // A chunk of b has shape (kBlockK, kBlockN) == (8, 128).
    // A thread block has shape (kBlockSize, kBlockSize) == (16, 16).
    // Thus, each thread should load 4 elements from a and another 4 elements from b.
    __shared__ float smemA[kBlockM][kBlockK + kPadding] = {};
    __shared__ float semeB[kBlockK][kBlockN + kPadding] = {};

    // For chunk a (128, 8), each two threads load a whole row (of size 8).
    // For chunk b (8, 128), each 32 threads load a whole row (of size 128).
    int rowA = baseIdx >> 1;          // each two threads load a row
    int colA = (baseIdx & 1) << 2;    // column index of the 1st element to load.

    int rowB = baseIdx >> 5;          // each 32 threads load a row
    int colB = (baseIdx << 2) & 127;  // column index of the 1st element to load.

    int rowC = ((warpIdx >> 1 << 3) + (laneIdx & 3)) << 2;
    int colC = (((warpIdx & 1) << 4) + (laneIdx >> 2)) << 2;

    const float * __restrict__ baseA = a + baseY * dk;
    const float * __restrict__ baseB = a + baseX;
    float * __restrict__ baseC = c + (baseY + rowC) * dn + baseX + colC;

    float4 regA[kThreadM >> 2] = {};
    float4 regB[kThreadN >> 2] = {};
    float regC[kThreadM][kThreadN] = {};

    for (int k0 = 0; k0 < dk; k0 += kBlockK)
    {

    }

//    acc_t reg[kThreadM][kThreadN] = {0};
//
//    for (int k = 0; k < dk; ++k)
//    {
//        for (int m = 0; m < kThreadM; ++m)
//        {
//            for (int n = 0; n < kThreadN; ++n)
//            {
//                if (ty + m < dm && tx + n < dn)
//                {
//                    reg[m][n] += a[(ty + m) * dk + k] * b[k * dn + (tx + n)];
//                }
//            }
//        }
//    }
//
//    for (int m = 0; m < kThreadM; ++m)
//    {
//        for (int n = 0; n < kThreadN; ++n)
//        {
//            if (ty + m < dm && tx + n < dn)
//            {
//                c[(ty + m) * dn + (tx + n)] =
//                        alpha * reg[m][n] +
//                        beta * c[(ty + m) * dn + (tx + n)];
//            }
//        }
//    }
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
        return abs(a - b) < kEps;
    }

    static constexpr float kEps = 1e-3f;
};


template <bool kDebugOutput = false, typename acc_t>
void checkResult(const thrust::device_vector<acc_t> & result,
                 const thrust::device_vector<acc_t> & golden,
                 int dm,
                 int dn)
{
    if constexpr (kDebugOutput)
    {
        thrust::host_vector<acc_t> a = result;
        thrust::host_vector<acc_t> b = golden;
        printf("Result:\n");
        for (int i = 0, k = 0; i < 1; ++i)
        {
            for (int j = 0; j < dn; ++j, ++k)
            {
                printf("%f ", a[k]);
            }
            printf("\n");
        }
        printf("\n\n");
        printf("Ground truth:\n");
        for (int i = 0, k = 0; i < 1; ++i)
        {
            for (int j = 0; j < dn; ++j, ++k)
            {
                printf("%f ", b[k]);
            }
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
    ///                    Disenable when debugging output.
    constexpr int kDup = 1;
    constexpr bool kRandInput = false;

    // Problem setting.
    int m = 1024;
    int n = 1024;
    int k = 1024;
    float alpha = 1.0f;
    float beta = 1.0f;
    thrust::host_vector<float> h_a(m * k, 1.0f);
    thrust::host_vector<float> h_b(k * n, 1.0f);
    thrust::host_vector<float> h_c(m * n, 0.0f);

    if constexpr (kRandInput)
    {
        unsigned seed = std::random_device()();
        std::default_random_engine e(seed);
        std::normal_distribution<float> d(0.0f, 1.0f);
        auto g = [&e, &d]() -> float { return d(e); };
        std::generate(h_a.begin(), h_a.end(), g);
        std::generate(h_b.begin(), h_b.end(), g);
        std::generate(h_c.begin(), h_c.end(), g);
    }

    thrust::device_vector<float> golden_c(m * n);
    thrust::device_vector<float> d_a = h_a;
    thrust::device_vector<float> d_b = h_b;
    thrust::device_vector<float> d_c = h_c;

    // Compute ground truth with cuBLAS.
    gemmCublas(thrust::raw_pointer_cast(d_a.data()),
               thrust::raw_pointer_cast(d_b.data()),
               thrust::raw_pointer_cast(d_c.data()),
               m,
               n,
               k,
               alpha,
               beta);
    golden_c = d_c;

    constexpr int kBlockSize = 16;
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kBlockK = 8;
    constexpr dim3 block(kBlockSize, kBlockSize);
    dim3 grid((n + kBlockN - 1) / kBlockN, (m + kBlockM - 1) / kBlockM);

    float ms;
    cudaEvent_t ss, ee;
    CUDA_CHECK(cudaEventCreate(&ss));
    CUDA_CHECK(cudaEventCreate(&ee));

    // Naive
    if constexpr (1 < kDup)
    {
        gemmNaive<kBlockSize, kBlockM, kBlockN, kBlockK><<<grid, block>>>(
                thrust::raw_pointer_cast(d_a.data()),
                thrust::raw_pointer_cast(d_b.data()),
                thrust::raw_pointer_cast(d_c.data()),
                m,
                n,
                k,
                alpha,
                beta
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    d_c = h_c;
    CUDA_CHECK(cudaEventRecord(ss));

    for (int dup = 0; dup < kDup; ++dup)
    {
        gemmNaive<kBlockSize, kBlockM, kBlockN, kBlockK><<<grid, block>>>(
            thrust::raw_pointer_cast(d_a.data()),
            thrust::raw_pointer_cast(d_b.data()),
            thrust::raw_pointer_cast(d_c.data()),
            m,
            n,
            k, 
            alpha,
            beta
        );
    }

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));

    std::printf("gemmNaive: ");
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    std::printf("took %f ms, ", ms / kDup);

    if constexpr (1 == kDup)
    {
        checkResult(d_c, golden_c, m, n);
    }
    else
    {
        std::printf("\n");
    }

    // Free cuda events.
    cudaEventDestroy(ss);
    cudaEventDestroy(ee);

    return EXIT_SUCCESS;
}