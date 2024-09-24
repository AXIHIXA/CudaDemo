#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


/// Naive CPU GEMM kernel calculating alpha * (A @ B) + beta * C.
/// Just for demo purpose. Do not call this method. It's SUPER SLOW!
/// \param A: [in] dtype=in_t, shape=(dm, dk) \n
/// \param B: [in] dtype=in_t, shape=(dk, dn) \n
/// \param C: [inout] dtype=out_t, shape=(dm, dn)
template <typename in_t, typename acc_t>
void gemmCpuNaive(const in_t * __restrict__ a,
                  const in_t * __restrict__ b,
                  acc_t * __restrict__ c,
                  int dm,
                  int dn,
                  int dk,
                  in_t alpha,
                  in_t beta)
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
/// \param A: [in] dtype=in_t, shape=(dm, dk) \n
/// \param B: [in] dtype=in_t, shape=(dk, dn) \n
/// \param C: [inout] dtype=out_t, shape=(dm, dn)
template <int kBlockSize, int kBlockM, int kBlockN, int kBlockK, typename in_t, typename acc_t>
__global__ void gemmNaive(const in_t * __restrict__ a,
                          const in_t * __restrict__ b,
                          acc_t * __restrict__ c,
                          int dm,
                          int dn,
                          int dk,
                          in_t alpha,
                          in_t beta)
{
    constexpr int kThreadM = kBlockM / kBlockSize;
    constexpr int kThreadN = kBlockN / kBlockSize;

    int tx = (blockIdx.x * blockDim.x + threadIdx.x) * kThreadM;
    int ty = (blockIdx.y * blockDim.y + threadIdx.y) * kThreadN;

    acc_t reg[kThreadM * kThreadN] = {};

    for (int i = 0; i < dk; i++)
    {
        for (int m = 0; m < kThreadM; m++)
        {
            for (int n = 0; n < kThreadN; n++)
            {
                reg[m * kThreadN + n] += a[(tx + m) * dk + i] * b[i * dn + ty + n];
            }
        }
    }

    for (int m = 0; m < kThreadM; m++)
    {
        for (int n = 0; n < kThreadN; n++)
        {
            // Multiply alpha here to reduce the alpha cal num.
            c[(tx + m) * dn + ty + n] =
                alpha * reg[m * kThreadN + n] +
                beta * c[(tx + m) * dn + ty + n];
        }
    }
}


/// GEMM kernel with cuBLAS, calculating alpha * (A @ B) + beta * C.
/// Used for ground truth calculation.
/// \param A: [in] dtype=float, shape=(dm, dk) \n
/// \param B: [in] dtype=float, shape=(dk, dn) \n
/// \param C: [inout] dtype=float, shape=(dm, dn)
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
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    CUBLAS_CHECK(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            dm,
            dn,
            dk,
            &alpha,
            a,
            CUDA_R_32F,
            dm,
            b,
            CUDA_R_32F,
            dk,
            &beta,
            c,
            CUDA_R_32F,
            dm,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT
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

    static constexpr float kEps = 1e-4f;
};


template <typename acc_t>
void checkResult(const thrust::device_vector<acc_t> & result, const thrust::device_vector<acc_t> & golden)
{
    bool resultIsCorrect = thrust::equal(thrust::device, result.cbegin(), result.cend(), golden.cbegin(), Equal<acc_t>());
    std::printf("Result is %s\n\n", resultIsCorrect ? "correct." : "WRONG!!!");
}


int main(int argc, char * argv[])
{
    int m = 1024;
    int n = 1024;
    int k = 1024;
    thrust::host_vector<float> h_a(m * k, 1.0f);
    thrust::host_vector<float> h_b(k * n, 1.0f);
    thrust::host_vector<float> h_c(m * n, 0.0f);
    thrust::device_vector<float> golden_c(m * n, 0.0f);
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
               1.0f,
               1.0f);
    golden_c = d_c;
    d_c = h_c;

    #ifdef NDEBUG
    constexpr int kDup = 1;
//    #define WARMUP
    #else
    constexpr int kDup = 1;
    #endif  // NDEBUG

    constexpr int kBlockSize = 16;
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kBlockK = 8;
    constexpr dim3 block(kBlockSize, kBlockSize);
    dim3 grid((m + kBlockM - 1) / kBlockM, (n + kBlockN - 1) / kBlockN);

    float ms;
    cudaEvent_t ss, ee;
    CUDA_CHECK(cudaEventCreate(&ss));
    CUDA_CHECK(cudaEventCreate(&ee));

    // Naive
    #ifdef WARMUP
    gemmNaive<kBlockSize, kBlockM, kBlockN, kBlockK><<<grid, block>>>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()),
        m,
        n,
        k,
        1.0f,
        1.0f
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    #endif  // WARMUP

    thrust::fill(thrust::device, d_c.begin(), d_c.end(), 0.0f);
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
            1.0f,
            1.0f
        );
    }

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));

    std::printf("gemmNaive: ");
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    std::printf("took %f ms, ", ms / kDup);
    checkResult(d_c, golden_c);

    // Free cuda events.
    cudaEventDestroy(ss);
    cudaEventDestroy(ee);

    return EXIT_SUCCESS;
}