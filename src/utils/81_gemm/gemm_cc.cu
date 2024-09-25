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
/// \param[in] A: shape=(dm, dk)
/// \param[in] B: shape=(dk, dn)
/// \param[in/out] C: shape=(dm, dn)
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


/// Naive GEMM kernel calculating alpha * (A @ B) + beta * C.
/// Used for baseline.
/// Each thread block has (kBlockSize, kBlockSize) threads,
/// Computing a square of (kBlockM, kBlockN) elements in output matrix.
/// Thus each thread computes a square of (kBlockM, kBlockN) / kBlockSize elements.
/// \param[in] A: shape=(dm, dk)
/// \param[in] B: shape=(dk, dn)
/// \param[in/out] C: shape=(dm, dn)
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

//    for (int i = 0; i < kThreadM; ++i)
//    {
//        for (int j = 0; j < kThreadN; ++j)
//        {
//            if (ty + i < dm && tx + j < dn && reg[i][j] != 512)
//            {
//                printf("(%d %d)\n", ty + i, tx + j);
//            }
//        }
//    }

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


/// GEMM kernel for fp32 with cuBLAS, calculating alpha * (A @ B) + beta * C.
/// Used for ground truth calculation.
/// Since cuBLAS matrices are in COLUMN-MAJOR:
/// Note that matrix A stored in row-major EQUALS A.T stored in column major.
/// We just let cuBLAS compute B.T @ A.T.
/// the result is (AB).T in column major, which equals AB in row major!
/// And, B.T and A.T (in column major) equals B and A (in row major).
/// So just pass in B and A (in row major, not transposed) and corresponding shapes (transposed).
/// \param[in] A: shape=(dm, dk)
/// \param[in] B: shape=(dk, dn)
/// \param[in/out] C: shape=(dm, dn)
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


template <typename acc_t>
void checkResult(const thrust::device_vector<acc_t> & result,
                 const thrust::device_vector<acc_t> & golden,
                 int dm,
                 int dn)
{
    #define DEBUG_OUTPUT
    #ifdef DEBUG_OUTPUT
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
    #endif  // DEBUG_OUTPUT

    bool resultIsCorrect = thrust::equal(thrust::device, result.cbegin(), result.cend(), golden.cbegin(), Equal<acc_t>());
    std::printf("Result is %s\n\n", resultIsCorrect ? "correct." : "WRONG!!!");
}


int main(int argc, char * argv[])
{
    int m = 1024;
    int n = 2048;
    int k = 512;
    float alpha = 1.0f;
    float beta = 1.0f;
    thrust::host_vector<float> h_a(m * k, 1.0f);
    thrust::host_vector<float> h_b(k * n, 1.0f);
    thrust::host_vector<float> h_c(m * n, 0.0f);

    constexpr bool kRandInput = true;

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

    // Switch for debugging output correctness.
    // Set to 1 to debug output (kernel only launched once) and results will be checked.
    // Set to values greater than 1 to profile.
    // In the latter case, results will NOT be checked because it's in-place GEMM.
    // We do not dispatch by build type because we have -G flag for Debug builds
    // (that's for debugging runtime errors).
    constexpr int kDup = 100;

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

    // Free cuda events.
    cudaEventDestroy(ss);
    cudaEventDestroy(ee);

    return EXIT_SUCCESS;
}