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
                float beta,
                cublasHandle_t handle)
{
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
/// Utilizes SMEM (does NOT handle bank conflicts),
/// and vectorized loads/stores (GMEM <-> REG <-> SMEM) via float4.
/// \n\n
/// Each thread block has (kBlockSize, kBlockSize) threads,
/// Computing a square of (kBlockM, kBlockN) elements in output matrix.
/// Thus each thread computes a square of (kBlockM, kBlockN) / kBlockSize elements.
/// \n\n
/// It's reasonable to tile by dimension K,
/// as each thread block computes by kBlockM * kBlockN * dk,
/// and accesses GMEM by kBlockM * dk + kBlockN * dk,
/// the compute-memory ratio is 1 / ( 1/kBlockM + 1/kBlockN ),
/// which is irrelevant with dimension K.
/// \n\n
/// Moreover, we take the "outer product" way to accumulate elements in output.
/// Each time a thread block loads
/// (kBlockM, kBlockK) elements from A,
/// (kBlockK, kBlockN) elements from B,
/// and this sliding window slides in a sequential manner.
/// \param[in] A      shape=(dm, dk)
/// \param[in] B      shape=(dk, dn)
/// \param[in/out] C  shape=(dm, dn)
template <int kBlockSize, int kBlockM, int kBlockN, int kBlockK>
__global__ void gemmSmem(const float * __restrict__ A,
                         const float * __restrict__ B,
                         float * __restrict__ C,
                         int dm,
                         int dn,
                         int dk,
                         float alpha,
                         float beta)
{
    static_assert(kBlockSize == 16 &&
                  kBlockM == 128 && (kBlockM % kBlockSize == 0) &&
                  kBlockN == 128 && (kBlockN % kBlockSize == 0) &&
                  kBlockK == 8,
                  "At present, we only tested the specified combination.");

    // The x-span (N) and y-span (M) of this thread.
    // This thread computes (kThreadM, kThreadN) elements in the output matrix shaped (dm, dn).
    constexpr int kThreadM = kBlockM / kBlockSize;
    constexpr int kThreadN = kBlockN / kBlockSize;

    // Top-left corner of this thread block's elements in C.
    int bx = blockIdx.x * kBlockN;
    int by = blockIdx.y * kBlockM;

    // Indexes.
    int tid = threadIdx.y * kBlockSize + threadIdx.x;  // Index of this thread in the parent thread block.
    int laneIdx = tid % 32;
    int warpIdx = tid / 32;

    // Read A and B chunks into smem.
    // A chunk has shape (kBlockM, kBlockK) == (128, 8).
    // B chunk has shape (kBlockK, kBlockN) == (8, 128).
    // While a hread block has shape (kBlockSize, kBlockSize) == (16, 16).
    // Thus, each thread should load a float4 from A, and another float4 from B.
    __shared__ float subA[kBlockM * kBlockK];
    __shared__ float subB[kBlockK * kBlockN];

    // Indices of the 1st element in A and B handled by this thread block.
    const float * __restrict__ baseA = A + by * dk;
    const float * __restrict__ baseB = B + bx;

    // For chunk A (128, 8), each two threads load a whole row (of size 8).
    // Chunk A GMEM -> SMEM pattern:
    //  baseA |  [0 1 2 3] [4 5 6 7] <- colA
    // -------+---------------------
    //   0    |     00        01
    //   1    |     02        03
    //   2    |     04        05
    //   ...  |     ...       ...
    //  rowA
    // We will be loading rows but storing SMEM transposed (in columns),
    // Thread #0 will conflict with thread #1, and $2 with $3, ...
    int rowA = (tid * 4) / kBlockK;  // each two threads load a row of 8 floats.
    int colA = (tid * 4) % kBlockK;  // column index of the 1st float to load, colA is a multiple of 4.

    // For chunk B (8, 128), each 32 threads load a whole row (of size 128).
    int rowB = (tid * 4) / kBlockN;  // each 32 threads load a row of 128 floats.
    int colB = (tid * 4) % kBlockN;  // column index of the 1st float to load, colB is a multiple of 4.

    // Region-of-C to write.
    // Warp-side view:
    // All row/col indices offset by 8 per thread.
    // So this pattern will have SMEM LOAD conflicts.
    //      |  0    8
    // -----+-----------
    //    0 | 00   16
    //    8 | 01   17
    //   16 | 02   18
    //  ... | ...  ...
    //  120 | 15   31
    // A thread will handle a 8x8 submatrix in C.
    // Block-side view:
    //     baseC  [ 0  16]  [17  31]  ... [96 111]  [112 127]
    // [  0 127]  [Warp 0]  [Warp 1]  ... [Warp 6]  [Warp  7]

    // Index of top-left corner of the region in C, handled by this thread.
    // Note that y-span uses threadIdx.x, (i.e., threadIdx is transposed in C).
    // This is the desired behavior.
    // E.g., block (0, 0) thread (1, 0) reads a column in chunk A (row in subA), spanning from rows 8~15.
    // And reads a row in chunk B, spanning from columns 0~7.
    // These elements contribute to 8x8 block, but threadIdx.x contributes to ROW index, and threadIdx.y to COLUMN index!
    float * __restrict__ baseC = C + (by + threadIdx.x * kThreadM) * dn + bx + threadIdx.y * kThreadN;

    // Intermediate registers for this thread's 8x8 elements.
    float4 regA[kThreadM / 4] = {};
    float4 regB[kThreadN / 4] = {};

    // Intermediate register for A @ B.
    float c[kThreadM * kThreadN] = {};

    for (int i = 0; i < dk; i += kBlockK)
    {
        // Each thread loads its float4 from matrix A into SMEM,
        // but stores in COLUMN-major, i.e., transposed.
        // That's because subA will be accessed in column major afterward
        // (the outer loop only executes for kBlockK == 8 times,
        // so chunk A must be accessed in column major.)
        // Note that this pattern has bank conflicts on stores to subA.
        // Each consecutive 2 threads have bank conflicts.
        // E.g., Block (0, 0), threads (0, 0), (1, 0), (2, 0), (3, 0), ...
        // Indices (bank): 0 128(0) 128x2(0) 128x3(0),       128x4(0) 128x5(0) 128x6(0) 128x7(0),
        //                 1 1+128(1) 1+128x2(1) 1+128x3(1), 1+128x4(1) 1+128x5(1) 1+128x6(1) 1+128x7(1),
        // Each two consecutive threads have bank conflicts.
        regA[0] = *reinterpret_cast<const float4 *>(baseA + i + rowA * dk + colA);
        subA[rowA + colA * kBlockM] = regA[0].x;
        subA[rowA + (colA + 1) * kBlockM] = regA[0].y;
        subA[rowA + (colA + 2) * kBlockM] = regA[0].z;
        subA[rowA + (colA + 3) * kBlockM] = regA[0].w;

        // Each thread loads its float4 from matrix B into smem.
        // Naturally no bank conflict here.
        regB[0] = *reinterpret_cast<const float4 *>(baseB + (i + rowB) * dn + colB);
        *reinterpret_cast<float4 *>(&subB[tid * 4]) = regB[0];

        // So all data needed for this tile on dim K are loaded into SMEM.
        __syncthreads();

        #pragma unroll
        for (int ii = 0; ii < kBlockK; ++ii)
        {
            // Load subA by row. Note that subA stores A-chunk in column major (transposed).
            // So all threads in this thread block are bringing into register a whole column in A-chunk.
            // Each threads holds 8 floats in a column.
            // Note that we only use threadIdx.x here, it's on purpose.
            // There are multiple C-blocks requiring the same rows in A!

            // Also note that the y-span for this block's C chunk is kBlockN == 128,
            // each thread handles x-span of kBlockK == 8,
            // so each element in subA should be read by 128/8==16 threads,
            // so that the whole y-span in C chunk is covered.

            // Also note that this pattern has bank conflicts on loads from subA.
            // Each consecutive 8 threads have bank conflicts.
            // E.g., Block (0, 0), Threads (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), ...
            // Indices (bank): 0, 8, 16, 24, 32(0), ...
            regA[0] = *reinterpret_cast<float4 *>(&subA[ii * kBlockM + threadIdx.x * kThreadM]);
            // E.g., Block (0, 0), Threads (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), ...
            // Indices (bank): 4, 12, 20, 28, 36(4), ...
            regA[1] = *reinterpret_cast<float4 *>(&subA[ii * kBlockM + threadIdx.x * kThreadM + 4]);

            // Load subB by row.
            // All threads in this thread block are bringing into register a whole row in B-chunk.
            // Each thread holds 8 floats in a row.
            // Naturally no bank conflict here (threads in the same phase read the same address).
            regB[0] = *reinterpret_cast<float4 *>(&subB[ii * kBlockN + threadIdx.y * kThreadN]);
            regB[1] = *reinterpret_cast<float4 *>(&subB[ii * kBlockN + threadIdx.y * kThreadN + 4]);

            // c is a kThreadM * kThreadN (8x8) block.
            //     0                 3   4                 7
            // 0   regA[0].x * regB[0]   regA[0].x * regB[1]
            // 1   regA[0].y * regB[0]   regA[0].x * regB[1]
            // 2   regA[0].z * regB[0]   regA[0].x * regB[1]
            // 3   regA[0].w * regB[0]   regA[0].x * regB[1]
            // 4   regA[1].x * regB[0]   regA[1].x * regB[1]
            // 5   regA[1].y * regB[0]   regA[1].x * regB[1]
            // 6   regA[1].z * regB[0]   regA[1].x * regB[1]
            // 7   regA[1].w * regB[0]   regA[1].x * regB[1]

            #pragma unroll
            for (int cpi = 0; cpi < kThreadM / 4; ++cpi)
            {
                #pragma unroll
                for (int cpj = 0; cpj < kThreadN / 4; ++cpj)
                {
                    c[cpi * 4 * kThreadM + cpj * 4] += regA[cpi].x * regB[cpj].x;
                    c[cpi * 4 * kThreadM + cpj * 4 + 1] += regA[cpi].x * regB[cpj].y;
                    c[cpi * 4 * kThreadM + cpj * 4 + 2] += regA[cpi].x * regB[cpj].z;
                    c[cpi * 4 * kThreadM + cpj * 4 + 3] += regA[cpi].x * regB[cpj].w;

                    c[(cpi * 4 + 1) * kThreadM + cpj * 4] += regA[cpi].y * regB[cpj].x;
                    c[(cpi * 4 + 1) * kThreadM + cpj * 4 + 1] += regA[cpi].y * regB[cpj].y;
                    c[(cpi * 4 + 1) * kThreadM + cpj * 4 + 2] += regA[cpi].y * regB[cpj].z;
                    c[(cpi * 4 + 1) * kThreadM + cpj * 4 + 3] += regA[cpi].y * regB[cpj].w;

                    c[(cpi * 4 + 2) * kThreadM + cpj * 4] += regA[cpi].z * regB[cpj].x;
                    c[(cpi * 4 + 2) * kThreadM + cpj * 4 + 1] += regA[cpi].z * regB[cpj].y;
                    c[(cpi * 4 + 2) * kThreadM + cpj * 4 + 2] += regA[cpi].z * regB[cpj].z;
                    c[(cpi * 4 + 2) * kThreadM + cpj * 4 + 3] += regA[cpi].z * regB[cpj].w;

                    c[(cpi * 4 + 3) * kThreadM + cpj * 4] += regA[cpi].w * regB[cpj].x;
                    c[(cpi * 4 + 3) * kThreadM + cpj * 4 + 1] += regA[cpi].w * regB[cpj].y;
                    c[(cpi * 4 + 3) * kThreadM + cpj * 4 + 2] += regA[cpi].w * regB[cpj].z;
                    c[(cpi * 4 + 3) * kThreadM + cpj * 4 + 3] += regA[cpi].w * regB[cpj].w;
                }
            }

            __syncthreads();
        }
    }

    #pragma unroll
    for (int i = 0; i < kThreadM; ++i)
    {
        #pragma unroll
        for (int j = 0; j < kThreadN; j += 4)
        {
            *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[i * dn + j]);
            regA[0].x = regA[0].x * beta + alpha * c[i * kThreadM + j];
            regA[0].y = regA[0].y * beta + alpha * c[i * kThreadM + j + 1];
            regA[0].z = regA[0].z * beta + alpha * c[i * kThreadM + j + 2];
            regA[0].w = regA[0].w * beta + alpha * c[i * kThreadM + j + 3];
            *reinterpret_cast<float4 *>(&baseC[i * dn + j]) = *reinterpret_cast<float4 *>(&regA[0]);
        }
    }
}


/// GEMM kernel calculating alpha * (A @ B) + beta * C.
/// Utilizes SMEM,
/// eliminates SMEM STORE bank conflicts via padding,
/// and SMEM LOAD bank conflicts via transposed warp tiling.
/// Also uses vectorized loads/stores (GMEM <-> REG <-> SMEM) via float4.
/// \param[in] A      shape=(dm, dk)
/// \param[in] B      shape=(dk, dn)
/// \param[in/out] C  shape=(dm, dn)
template <int kBlockSize, int kBlockM, int kBlockN, int kBlockK, int kPadding = 4>
__global__ void gemmSmemPadTransposedThread(const float * __restrict__ A,
                                            const float * __restrict__ B,
                                            float * __restrict__ C,
                                            int dm,
                                            int dn,
                                            int dk,
                                            float alpha,
                                            float beta)
{
    // Pad by 4 to align float4 loads/stores.
    static_assert(kBlockSize == 16 &&
                  kBlockM == 128 && (kBlockM % kBlockSize == 0) &&
                  kBlockN == 128 && (kBlockN % kBlockSize == 0) &&
                  kBlockK == 8 &&
                  kPadding == 4,
                  "At present, we only tested the specified combination.");

    // The x-span (N) and y-span (M) of this thread.
    // This thread computes (kThreadM, kThreadN) elements in the output matrix shaped (dm, dn).
    constexpr int kThreadM = kBlockM / kBlockSize;
    constexpr int kThreadN = kBlockN / kBlockSize;

    // Top-left corner of this thread block's elements in C.
    int bx = blockIdx.x * kBlockN;
    int by = blockIdx.y * kBlockM;

    // Indexes.
    int tid = threadIdx.y * kBlockSize + threadIdx.x;  // Index of this thread in the parent thread block.
    int laneIdx = tid % 32;
    int warpIdx = tid / 32;

    // Read A and B chunks into smem.
    // A chunk has shape (kBlockM, kBlockK) == (128, 8).
    // B chunk has shape (kBlockK, kBlockN) == (8, 128).
    // While a hread block has shape (kBlockSize, kBlockSize) == (16, 16).
    // Thus, each thread should load a float4 from A, and another float4 from B.
    // Padding resolves bank conflicts on SMEM stores.
    constexpr int kLdSubA = kBlockM + kPadding;
    __shared__ float subA[kLdSubA * kBlockK];
    __shared__ float subB[kBlockK * kBlockN];

    // Indices of the 1st element in A and B handled by this thread block.
    const float * __restrict__ baseA = A + by * dk;
    const float * __restrict__ baseB = B + bx;

    // For chunk A (128, 8), each two threads load a whole row (of size 8).
    int rowA = (tid * 4) / kBlockK;  // each two threads load a row of 8 floats.
    int colA = (tid * 4) % kBlockK;  // column index of the 1st float to load, colA is a multiple of 4.

    // For chunk B (8, 128), each 32 threads load a whole row (of size 128).
    int rowB = (tid * 4) / kBlockN;  // each 32 threads load a row of 128 floats.
    int colB = (tid * 4) % kBlockN;  // column index of the 1st float to load, colB is a multiple of 4.

    // Index of C.
    // Padding resolves bank conflicts on SMEM stores.
    // The special transposed warp tiling pattern resolves bank conflicts on SMEM loads.
    // Warp-side view:
    // All row/col indices offset by 4 per thread.
    //    |  0   4   8  12  16  20  24  28
    // ---+-------------------------------
    //  0 | 00  04  08  12  16  20  24  28
    //  4 | 01  05  09  13  17  21  25  29
    //  8 | 02  06  10  14  18  22  26  30
    // 12 | 03  07  11  15  19  23  27  31
    // A thread will no longer handle a 8x8 submatrix in C.
    // it handles four 4x4 submatrices located (top-left-corner's linear index in C)
    // at baseC, baseC + 32, baseC + 16 * dn, baseC + 16 * dn + 32, in a loop[1...4].
    // Block-side view:
    //     baseC  [ 0  31]  [32  63]  [64  95]  [96 127]
    //  [ 0  15]  [Warp 0]  [Warp  ]  [Warp 1]  [Warp  ]
    //  [16  31]  [Warp  ]  [Warp  ]  [Warp  ]  [Warp  ]
    //  [32  47]  [Warp 2]  [Warp  ]  [Warp 3]  [Warp  ]
    //  [48  63]  [Warp  ]  [Warp  ]  [Warp  ]  [Warp  ]
    //  [64  79]  [Warp 4]  [Warp  ]  [Warp 5]  [Warp  ]
    //  [80  95]  [Warp  ]  [Warp  ]  [Warp  ]  [Warp  ]
    //  [96 111]  [Warp 6]  [Warp  ]  [Warp 7]  [Warp  ]
    // [112 127]  [Warp  ]  [Warp  ]  [Warp  ]  [Warp  ]
    int rowC = ((warpIdx >> 1 << 3) + (laneIdx & 3)) << 2;
    int colC = (((warpIdx & 1) << 4) + (laneIdx >> 2)) << 2;
    float * __restrict__ baseC = C + (by + rowC) * dn + bx + colC;

    // Intermediate registers for this thread's 8x8 elements.
    float4 regA[kThreadM / 4] = {};
    float4 regB[kThreadN / 4] = {};

    // Intermediate results.
    float c[kThreadM * kThreadN] = {};

    for (int i = 0; i < dk; i += kBlockK)
    {
        // Each thread loads its float4 from matrix A into smem (but stores in COLUMN-major).
        // That's because smemA will be accessed in column major afterward.
        // Since we're writing SMEM in column major, we need to pad column to eliminate bank conflicts.
        // Also note that the y-span for this block's C chunk is kBlockN == 128,
        // each thread handles x-span of kBlockK == 8,
        // so each element in subA should be read by 128/8==16 threads, so that the whole y-span in C chunk is covered.
        regA[0] = *reinterpret_cast<const float4 *>(baseA + i + rowA * dk + colA);
        subA[rowA + colA * kLdSubA] = regA[0].x;
        subA[rowA + (colA + 1) * kLdSubA] = regA[0].y;
        subA[rowA + (colA + 2) * kLdSubA] = regA[0].z;
        subA[rowA + (colA + 3) * kLdSubA] = regA[0].w;

        // Each thread loads its float4 from matrix B into smem.
        // Because the SMEM stores are coalesced, there's natually no bank conflict.
        regB[0] = *reinterpret_cast<const float4 *>(baseB + (i + rowB) * dn + colB);
        *reinterpret_cast<float4 *>(&subB[tid * 4]) = regB[0];

        // So all data needed for this tile on dim K are loaded into SMEM.
        __syncthreads();

        // So the SMEM stores are indentical to naive SMEM kernel, except the padding.

        #pragma unroll
        for (int ii = 0; ii < kBlockK; ++ii)
        {
            // No bank conflict this time.
            // But since regA[0...1] does not hold consecutive 8 floats,
            // the results need to be stored to somewhere else (strided...)
            // E.g., block (0, 0), thread (0, 0), (1, 0), (2, 0), (3, 0), ...
            // subA   0   4   8  12  0  4  8  12 (broadcast, no bank conflict)
            regA[0] = *reinterpret_cast<float4 *>(&subA[ii * kLdSubA + rowC]);
            // subA  16  20  24  28  16  20  24  28 (broadcast, no bank conflict)
            regA[1] = *reinterpret_cast<float4 *>(&subA[ii * kLdSubA + rowC + 16]);
            // subB   0   0   0   0   4   4   4   4 (broadcast, no bank conflict)
            regB[0] = *reinterpret_cast<float4 *>(&subB[ii * kBlockN + colC]);
            // subB  32(0) 32(0) 32(0) 32(0) 36(4) 36(4) 36(4) 36(4) (broadcast, no bank conflict)
            regB[1] = *reinterpret_cast<float4 *>(&subB[ii * kBlockN + colC + 32]);

            // c is a kThreadM * kThreadN (8x8) block, SAME AS SmemNaive.
            //     0                 3   4                 7
            // 0   regA[0].x * regB[0]   regA[0].x * regB[1]
            // 1   regA[0].y * regB[0]   regA[0].x * regB[1]
            // 2   regA[0].z * regB[0]   regA[0].x * regB[1]
            // 3   regA[0].w * regB[0]   regA[0].x * regB[1]
            // 4   regA[1].x * regB[0]   regA[1].x * regB[1]
            // 5   regA[1].y * regB[0]   regA[1].x * regB[1]
            // 6   regA[1].z * regB[0]   regA[1].x * regB[1]
            // 7   regA[1].w * regB[0]   regA[1].x * regB[1]

            #pragma unroll
            for (int cpi = 0; cpi < kThreadM / 4; ++cpi)
            {
                #pragma unroll
                for (int cpj = 0; cpj < kThreadN / 4; ++cpj)
                {
                    c[cpi * 4 * kThreadM + cpj * 4] += regA[cpi].x * regB[cpj].x;
                    c[cpi * 4 * kThreadM + cpj * 4 + 1] += regA[cpi].x * regB[cpj].y;
                    c[cpi * 4 * kThreadM + cpj * 4 + 2] += regA[cpi].x * regB[cpj].z;
                    c[cpi * 4 * kThreadM + cpj * 4 + 3] += regA[cpi].x * regB[cpj].w;

                    c[(cpi * 4 + 1) * kThreadM + cpj * 4] += regA[cpi].y * regB[cpj].x;
                    c[(cpi * 4 + 1) * kThreadM + cpj * 4 + 1] += regA[cpi].y * regB[cpj].y;
                    c[(cpi * 4 + 1) * kThreadM + cpj * 4 + 2] += regA[cpi].y * regB[cpj].z;
                    c[(cpi * 4 + 1) * kThreadM + cpj * 4 + 3] += regA[cpi].y * regB[cpj].w;

                    c[(cpi * 4 + 2) * kThreadM + cpj * 4] += regA[cpi].z * regB[cpj].x;
                    c[(cpi * 4 + 2) * kThreadM + cpj * 4 + 1] += regA[cpi].z * regB[cpj].y;
                    c[(cpi * 4 + 2) * kThreadM + cpj * 4 + 2] += regA[cpi].z * regB[cpj].z;
                    c[(cpi * 4 + 2) * kThreadM + cpj * 4 + 3] += regA[cpi].z * regB[cpj].w;

                    c[(cpi * 4 + 3) * kThreadM + cpj * 4] += regA[cpi].w * regB[cpj].x;
                    c[(cpi * 4 + 3) * kThreadM + cpj * 4 + 1] += regA[cpi].w * regB[cpj].y;
                    c[(cpi * 4 + 3) * kThreadM + cpj * 4 + 2] += regA[cpi].w * regB[cpj].z;
                    c[(cpi * 4 + 3) * kThreadM + cpj * 4 + 3] += regA[cpi].w * regB[cpj].w;
                }
            }

            __syncthreads();
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[i * dn]);
        regA[0].x = regA[0].x * beta + alpha * c[i * kThreadN];
        regA[0].y = regA[0].y * beta + alpha * c[1 + i * kThreadN];
        regA[0].z = regA[0].z * beta + alpha * c[2 + i * kThreadN];
        regA[0].w = regA[0].w * beta + alpha * c[3 + i * kThreadN];
        *reinterpret_cast<float4 *>(&baseC[i * dn]) = *reinterpret_cast<float4 *>(&regA[0]);

        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[i * dn + 32]);
        regA[0].x = regA[0].x * beta + alpha * c[4 + i * kThreadN];
        regA[0].y = regA[0].y * beta + alpha * c[5 + i * kThreadN];
        regA[0].z = regA[0].z * beta + alpha * c[6 + i * kThreadN];
        regA[0].w = regA[0].w * beta + alpha * c[7 + i * kThreadN];
        *reinterpret_cast<float4 *>(&baseC[i * dn + 32]) = *reinterpret_cast<float4 *>(&regA[0]);

        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[(i + 16) * dn]);
        regA[0].x = regA[0].x * beta + alpha * c[32 + i * kThreadN];
        regA[0].y = regA[0].y * beta + alpha * c[33 + i * kThreadN];
        regA[0].z = regA[0].z * beta + alpha * c[34 + i * kThreadN];
        regA[0].w = regA[0].w * beta + alpha * c[35 + i * kThreadN];
        *reinterpret_cast<float4 *>(&baseC[(i + 16) * dn]) = *reinterpret_cast<float4 *>(&regA[0]);

        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[(i + 16) * dn + 32]);
        regA[0].x = regA[0].x * beta + alpha * c[36 + i * kThreadN];
        regA[0].y = regA[0].y * beta + alpha * c[37 + i * kThreadN];
        regA[0].z = regA[0].z * beta + alpha * c[38 + i * kThreadN];
        regA[0].w = regA[0].w * beta + alpha * c[39 + i * kThreadN];
        *reinterpret_cast<float4 *>(&baseC[(i + 16) * dn + 32]) = *reinterpret_cast<float4 *>(&regA[0]);
    }
}


/// GEMM kernel calculating alpha * (A @ B) + beta * C.
/// Utilizes SMEM,
/// eliminates SMEM STORE bank conflicts via padding,
/// and SMEM LOAD bank conflicts via warp tiling (z-threads).
/// The only difference with TransposedThread is the rowC/colC indices.
/// Also uses vectorized loads/stores (GMEM <-> REG <-> SMEM) via float4.
/// \param[in] A      shape=(dm, dk)
/// \param[in] B      shape=(dk, dn)
/// \param[in/out] C  shape=(dm, dn)
template <int kBlockSize, int kBlockM, int kBlockN, int kBlockK, int kPadding = 4>
__global__ void gemmSmemPadZThread(const float * __restrict__ A,
                                   const float * __restrict__ B,
                                   float * __restrict__ C,
                                   int dm,
                                   int dn,
                                   int dk,
                                   float alpha,
                                   float beta)
{
    // Pad by 4 to align float4 loads/stores.
    static_assert(kBlockSize == 16 &&
                  kBlockM == 128 && (kBlockM % kBlockSize == 0) &&
                  kBlockN == 128 && (kBlockN % kBlockSize == 0) &&
                  kBlockK == 8 &&
                  kPadding == 4,
                  "At present, we only tested the specified combination.");

    // The x-span (N) and y-span (M) of this thread.
    // This thread computes (kThreadM, kThreadN) elements in the output matrix shaped (dm, dn).
    constexpr int kThreadM = kBlockM / kBlockSize;
    constexpr int kThreadN = kBlockN / kBlockSize;

    // Top-left corner of this thread block's elements in C.
    int bx = blockIdx.x * kBlockN;
    int by = blockIdx.y * kBlockM;

    // Indexes.
    int tid = threadIdx.y * kBlockSize + threadIdx.x;  // Index of this thread in the parent thread block.
    int laneIdx = tid % 32;
    int warpIdx = tid / 32;

    // Read A and B chunks into smem.
    // A chunk has shape (kBlockM, kBlockK) == (128, 8).
    // B chunk has shape (kBlockK, kBlockN) == (8, 128).
    // While a hread block has shape (kBlockSize, kBlockSize) == (16, 16).
    // Thus, each thread should load a float4 from A, and another float4 from B.
    // Padding resolves bank conflicts on SMEM stores.
    constexpr int kLdSubA = kBlockM + kPadding;
    __shared__ float subA[kLdSubA * kBlockK];
    __shared__ float subB[kBlockK * kBlockN];

    // Indices of the 1st element in A and B handled by this thread block.
    const float * __restrict__ baseA = A + by * dk;
    const float * __restrict__ baseB = B + bx;

    // For chunk A (128, 8), each two threads load a whole row (of size 8).
    int rowA = (tid * 4) / kBlockK;  // each two threads load a row of 8 floats.
    int colA = (tid * 4) % kBlockK;  // column index of the 1st float to load, colA is a multiple of 4.

    // For chunk B (8, 128), each 32 threads load a whole row (of size 128).
    int rowB = (tid * 4) / kBlockN;  // each 32 threads load a row of 128 floats.
    int colB = (tid * 4) % kBlockN;  // column index of the 1st float to load, colB is a multiple of 4.

    // Index of C.
    // Padding resolves bank conflicts on SMEM stores.
    // The special [z-shaped] (an alternative to TRANSPOSED) warp tiling pattern resolves bank conflicts on SMEM loads.
    // Warp-side view:
    // All row/col indices offset by 4 per thread.
    //    |  0   4    8  12   16  20   24  28
    // ---+--------+--------+-----------------
    //  0 | 00  02 | 04  06 | 08  10 | 12  14
    //  4 | 01  03 | 05  07 | 09  11 | 13  15
    //    |--------+--------+--------+--------
    //  8 | 16  18 | 20  22 | 24  26 | 28  30
    // 12 | 17  19 | 21  23 | 25  27 | 29  31
    // A thread will no longer handle a 8x8 submatrix in C.
    // it handles four 4x4 submatrices located (top-left-corner's linear index in C)
    // at baseC, baseC + 32, baseC + 16 * dn, baseC + 16 * dn + 32, in a loop[1...4].
    // Block-side view:
    //     baseC  [ 0  31]  [32  63]  [64  95]  [96 127]
    //  [ 0  15]  [Warp 0]  [Warp  ]  [Warp 1]  [Warp  ]
    //  [16  31]  [Warp  ]  [Warp  ]  [Warp  ]  [Warp  ]
    //  [32  47]  [Warp 2]  [Warp  ]  [Warp 3]  [Warp  ]
    //  [48  63]  [Warp  ]  [Warp  ]  [Warp  ]  [Warp  ]
    //  [64  79]  [Warp 4]  [Warp  ]  [Warp 5]  [Warp  ]
    //  [80  95]  [Warp  ]  [Warp  ]  [Warp  ]  [Warp  ]
    //  [96 111]  [Warp 6]  [Warp  ]  [Warp 7]  [Warp  ]
    // [112 127]  [Warp  ]  [Warp  ]  [Warp  ]  [Warp  ]
    int rowC = ((warpIdx >> 1 << 3) + ((laneIdx >> 4) << 1) + (laneIdx & 1)) << 2;
    int colC = (((warpIdx & 1) << 4) + ((laneIdx & 15) >> 1)) << 2;
    float * __restrict__ baseC = C + (by + rowC) * dn + bx + colC;

    // Intermediate registers for this thread's 8x8 elements.
    float4 regA[kThreadM / 4] = {};
    float4 regB[kThreadN / 4] = {};

    // Intermediate results.
    float c[kThreadM * kThreadN] = {};

    for (int i = 0; i < dk; i += kBlockK)
    {
        // Each thread loads its float4 from matrix A into smem (but stores in COLUMN-major).
        // That's because smemA will be accessed in column major afterward.
        // Since we're writing SMEM in column major, we need to pad column to eliminate bank conflicts.
        // Also note that the y-span for this block's C chunk is kBlockN == 128,
        // each thread handles x-span of kBlockK == 8,
        // so each element in subA should be read by 128/8==16 threads, so that the whole y-span in C chunk is covered.
        regA[0] = *reinterpret_cast<const float4 *>(baseA + i + rowA * dk + colA);
        subA[rowA + colA * kLdSubA] = regA[0].x;
        subA[rowA + (colA + 1) * kLdSubA] = regA[0].y;
        subA[rowA + (colA + 2) * kLdSubA] = regA[0].z;
        subA[rowA + (colA + 3) * kLdSubA] = regA[0].w;

        // Each thread loads its float4 from matrix B into smem.
        // Because the SMEM stores are coalesced, there's natually no bank conflict.
        regB[0] = *reinterpret_cast<const float4 *>(baseB + (i + rowB) * dn + colB);
        *reinterpret_cast<float4 *>(&subB[tid * 4]) = regB[0];

        // So all data needed for this tile on dim K are loaded into SMEM.
        __syncthreads();

        // So the SMEM stores are indentical to naive SMEM kernel, except the padding.

        #pragma unroll
        for (int ii = 0; ii < kBlockK; ++ii)
        {
            // No bank conflict this time.
            // But since regA[0...1] does not hold consecutive 8 floats,
            // the results need to be stored to somewhere else (strided...)
            // E.g., block (0, 0), thread (0, 0), (1, 0), (2, 0), (3, 0), ...
            // subA   0   4   8  12  0  4  8  12 (broadcast, no bank conflict)
            regA[0] = *reinterpret_cast<float4 *>(&subA[ii * kLdSubA + rowC]);
            // subA  16  20  24  28  16  20  24  28 (broadcast, no bank conflict)
            regA[1] = *reinterpret_cast<float4 *>(&subA[ii * kLdSubA + rowC + 16]);
            // subB   0   0   0   0   4   4   4   4 (broadcast, no bank conflict)
            regB[0] = *reinterpret_cast<float4 *>(&subB[ii * kBlockN + colC]);
            // subB  32(0) 32(0) 32(0) 32(0) 36(4) 36(4) 36(4) 36(4) (broadcast, no bank conflict)
            regB[1] = *reinterpret_cast<float4 *>(&subB[ii * kBlockN + colC + 32]);

            // c is a kThreadM * kThreadN (8x8) block, SAME AS SmemNaive.
            //     0                 3   4                 7
            // 0   regA[0].x * regB[0]   regA[0].x * regB[1]
            // 1   regA[0].y * regB[0]   regA[0].x * regB[1]
            // 2   regA[0].z * regB[0]   regA[0].x * regB[1]
            // 3   regA[0].w * regB[0]   regA[0].x * regB[1]
            // 4   regA[1].x * regB[0]   regA[1].x * regB[1]
            // 5   regA[1].y * regB[0]   regA[1].x * regB[1]
            // 6   regA[1].z * regB[0]   regA[1].x * regB[1]
            // 7   regA[1].w * regB[0]   regA[1].x * regB[1]

            #pragma unroll
            for (int cpi = 0; cpi < kThreadM / 4; ++cpi)
            {
                #pragma unroll
                for (int cpj = 0; cpj < kThreadN / 4; ++cpj)
                {
                    c[cpi * 4 * kThreadM + cpj * 4] += regA[cpi].x * regB[cpj].x;
                    c[cpi * 4 * kThreadM + cpj * 4 + 1] += regA[cpi].x * regB[cpj].y;
                    c[cpi * 4 * kThreadM + cpj * 4 + 2] += regA[cpi].x * regB[cpj].z;
                    c[cpi * 4 * kThreadM + cpj * 4 + 3] += regA[cpi].x * regB[cpj].w;

                    c[(cpi * 4 + 1) * kThreadM + cpj * 4] += regA[cpi].y * regB[cpj].x;
                    c[(cpi * 4 + 1) * kThreadM + cpj * 4 + 1] += regA[cpi].y * regB[cpj].y;
                    c[(cpi * 4 + 1) * kThreadM + cpj * 4 + 2] += regA[cpi].y * regB[cpj].z;
                    c[(cpi * 4 + 1) * kThreadM + cpj * 4 + 3] += regA[cpi].y * regB[cpj].w;

                    c[(cpi * 4 + 2) * kThreadM + cpj * 4] += regA[cpi].z * regB[cpj].x;
                    c[(cpi * 4 + 2) * kThreadM + cpj * 4 + 1] += regA[cpi].z * regB[cpj].y;
                    c[(cpi * 4 + 2) * kThreadM + cpj * 4 + 2] += regA[cpi].z * regB[cpj].z;
                    c[(cpi * 4 + 2) * kThreadM + cpj * 4 + 3] += regA[cpi].z * regB[cpj].w;

                    c[(cpi * 4 + 3) * kThreadM + cpj * 4] += regA[cpi].w * regB[cpj].x;
                    c[(cpi * 4 + 3) * kThreadM + cpj * 4 + 1] += regA[cpi].w * regB[cpj].y;
                    c[(cpi * 4 + 3) * kThreadM + cpj * 4 + 2] += regA[cpi].w * regB[cpj].z;
                    c[(cpi * 4 + 3) * kThreadM + cpj * 4 + 3] += regA[cpi].w * regB[cpj].w;
                }
            }

            __syncthreads();
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[i * dn]);
        regA[0].x = regA[0].x * beta + alpha * c[i * kThreadN];
        regA[0].y = regA[0].y * beta + alpha * c[1 + i * kThreadN];
        regA[0].z = regA[0].z * beta + alpha * c[2 + i * kThreadN];
        regA[0].w = regA[0].w * beta + alpha * c[3 + i * kThreadN];
        *reinterpret_cast<float4 *>(&baseC[i * dn]) = *reinterpret_cast<float4 *>(&regA[0]);

        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[i * dn + 32]);
        regA[0].x = regA[0].x * beta + alpha * c[4 + i * kThreadN];
        regA[0].y = regA[0].y * beta + alpha * c[5 + i * kThreadN];
        regA[0].z = regA[0].z * beta + alpha * c[6 + i * kThreadN];
        regA[0].w = regA[0].w * beta + alpha * c[7 + i * kThreadN];
        *reinterpret_cast<float4 *>(&baseC[i * dn + 32]) = *reinterpret_cast<float4 *>(&regA[0]);

        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[(i + 16) * dn]);
        regA[0].x = regA[0].x * beta + alpha * c[32 + i * kThreadN];
        regA[0].y = regA[0].y * beta + alpha * c[33 + i * kThreadN];
        regA[0].z = regA[0].z * beta + alpha * c[34 + i * kThreadN];
        regA[0].w = regA[0].w * beta + alpha * c[35 + i * kThreadN];
        *reinterpret_cast<float4 *>(&baseC[(i + 16) * dn]) = *reinterpret_cast<float4 *>(&regA[0]);

        *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[(i + 16) * dn + 32]);
        regA[0].x = regA[0].x * beta + alpha * c[36 + i * kThreadN];
        regA[0].y = regA[0].y * beta + alpha * c[37 + i * kThreadN];
        regA[0].z = regA[0].z * beta + alpha * c[38 + i * kThreadN];
        regA[0].w = regA[0].w * beta + alpha * c[39 + i * kThreadN];
        *reinterpret_cast<float4 *>(&baseC[(i + 16) * dn + 32]) = *reinterpret_cast<float4 *>(&regA[0]);
    }
}


/// GEMM kernel calculating alpha * (A @ B) + beta * C.
/// Utilizes SMEM,
/// eliminates SMEM bank conflicts (STORES AND LOADS) via warp tiling.
/// Also uses vectorized loads/stores (GMEM <-> REG <-> SMEM) via float4.
/// \param[in] A      shape=(dm, dk)
/// \param[in] B      shape=(dk, dn)
/// \param[in/out] C  shape=(dm, dn)
template <int kBlockSize, int kBlockM, int kBlockN, int kBlockK>
__global__ void gemmSmemWarpTile(const float * __restrict__ A,
                                 const float * __restrict__ B,
                                 float * __restrict__ C,
                                 int dm,
                                 int dn,
                                 int dk,
                                 float alpha,
                                 float beta)
{
    // Pad by 4 to align float4 loads/stores.
    static_assert(kBlockSize == 16 &&
                  kBlockM == 128 && (kBlockM % kBlockSize == 0) &&
                  kBlockN == 128 && (kBlockN % kBlockSize == 0) &&
                  kBlockK == 8,
                  "At present, we only tested the specified combination.");

    // The x-span (N) and y-span (M) of this thread.
    // This thread computes (kThreadM, kThreadN) elements in the output matrix shaped (dm, dn).
    constexpr int kThreadM = kBlockM / kBlockSize;
    constexpr int kThreadN = kBlockN / kBlockSize;

    // Top-left corner of this thread block's elements in C.
    int bx = blockIdx.x * kBlockN;
    int by = blockIdx.y * kBlockM;

    // Indexes.
    int tid = threadIdx.y * kBlockSize + threadIdx.x;  // Index of this thread in the parent thread block.
    int laneIdx = tid % 32;
    int warpIdx = tid / 32;

    // Read A and B chunks into smem.
    // A chunk has shape (kBlockM, kBlockK) == (128, 8).
    // B chunk has shape (kBlockK, kBlockN) == (8, 128).
    // While a hread block has shape (kBlockSize, kBlockSize) == (16, 16).
    // Thus, each thread should load a float4 from A, and another float4 from B.
    constexpr int kLdSubA = kBlockM;  // Eliminate bank conflicts by warp tiling.
    __shared__ float subA[kLdSubA * kBlockK];
    __shared__ float subB[kBlockK * kBlockN];

    // Indices of the 1st element in A and B handled by this thread block.
    const float * __restrict__ baseA = A + by * dk;
    const float * __restrict__ baseB = B + bx;

    // For chunk A (128, 8), each row (of size 8) needs two threads to load.
    // However, to avoid STORE bank conflicts, we do NOT let consecutive threads-of-twos to load a row.
    // Chunk A GMEM -> SMEM pattern:
    //  baseA |  [0 1 2 3] [4 5 6 7] <- colA
    // -------+---------------------
    //   0    |     00        32
    //   1    |     01        33
    //   ...  |     ...       ...
    //   31   |     31        63
    // -------+---------------------
    //   32   |     64        80
    //   33   |     65        33
    //   ...  |     ...       ...
    //   63   |     79        95
    //   ...  |     ...       ...
    //  rowA
    // We will STILL be loading rows but storing SMEM transposed (in columns).
    // But with this thread arrangement, we won't have SMEM STORE bank conflicts.
    // // int rowA = 32 * (warpIdx / 2) + laneIdx % 32;
    // // int colA = ((tid / 32) & 1) * 4;

    // The naive GMEM -> SMEM pattern.
    // This pattern has SMEM STORE bank conflicts but has slightly better performance that the no-bank-conflict version.
    // The cause should be the bank-conflict version has higher gld_efficiency.
    int rowA = (tid * 4) / kBlockK;  // each two threads load a row of 8 floats.
    int colA = (tid * 4) % kBlockK;  // column index of the 1st float to load, colA is a multiple of 4.

    // For chunk B (8, 128), each 32 threads load a whole row (of size 128).
    int rowB = (tid * 4) / kBlockN;  // each 32 threads load a row of 128 floats.
    int colB = (tid * 4) % kBlockN;  // column index of the 1st float to load, colB is a multiple of 4.

    // Index of C with warp tiling.
    // Warp-side view:
    // All row/col indices offset by 8 per thread.
    //    |  0   8  16  24  32  40  48  56
    // ---+-------------------------------
    //  0 | 00  04  08  12  16  20  24  28
    //  8 | 01  05  09  13  17  21  25  29
    // 16 | 02  06  10  14  18  22  26  30
    // 24 | 03  07  11  15  19  23  27  31
    // A thread handles a 8x8 submatrix in C.
    // Block-side view:
    //    baseC  [ 0  64]  [64 127]
    // [ 0  31]  [Warp 0]  [Warp 1]
    // [32  63]  [Warp 2]  [Warp 3]
    // [64  95]  [Warp 4]  [Warp 5]
    // [96 127]  [Warp 6]  [Warp 7]
    int rowC = ((warpIdx >> 1 << 2) + (laneIdx & 3)) << 3;
    int colC = (((warpIdx & 1) << 3) + (laneIdx >> 2)) << 3;
    float * __restrict__ baseC = C + (by + rowC) * dn + bx + colC;

    // Intermediate registers for this thread's 8x8 elements.
    float4 regA[kThreadM / 4] = {};
    float4 regB[kThreadN / 4] = {};

    // Intermediate results.
    float c[kThreadM * kThreadN] = {};

    for (int i = 0; i < dk; i += kBlockK)
    {
        // Warp-tiled SMEM STORE pattern as specified above.
        regA[0] = *reinterpret_cast<const float4 *>(baseA + i + rowA * dk + colA);
        subA[rowA + colA * kLdSubA] = regA[0].x;
        subA[rowA + (colA + 1) * kLdSubA] = regA[0].y;
        subA[rowA + (colA + 2) * kLdSubA] = regA[0].z;
        subA[rowA + (colA + 3) * kLdSubA] = regA[0].w;

        regB[0] = *reinterpret_cast<const float4 *>(baseB + (rowB + i) * dn + colB);
        *reinterpret_cast<float4 *>(&subB[tid * 4]) = regB[0];

        __syncthreads();

        #pragma unroll
        for (int ii = 0; ii < kBlockK; ++ii)
        {
            // Warp tiling.
            regA[0] = *reinterpret_cast<float4 *>(&subA[ii * kLdSubA + rowC]);
            regA[1] = *reinterpret_cast<float4 *>(&subA[ii * kLdSubA + rowC + 4]);
            regB[0] = *reinterpret_cast<float4 *>(&subB[ii * kBlockN + colC]);
            regB[1] = *reinterpret_cast<float4 *>(&subB[ii * kBlockN + colC + 4]);

            #pragma unroll
            for (int cpi = 0; cpi < kThreadM / 4; ++cpi)
            {
                #pragma unroll
                for (int cpj = 0; cpj < kThreadN / 4; ++cpj)
                {
                    c[cpi * 4 * kThreadM + cpj * 4] += regA[cpi].x * regB[cpj].x;
                    c[cpi * 4 * kThreadM + cpj * 4 + 1] += regA[cpi].x * regB[cpj].y;
                    c[cpi * 4 * kThreadM + cpj * 4 + 2] += regA[cpi].x * regB[cpj].z;
                    c[cpi * 4 * kThreadM + cpj * 4 + 3] += regA[cpi].x * regB[cpj].w;

                    c[(cpi * 4 + 1) * kThreadM + cpj * 4] += regA[cpi].y * regB[cpj].x;
                    c[(cpi * 4 + 1) * kThreadM + cpj * 4 + 1] += regA[cpi].y * regB[cpj].y;
                    c[(cpi * 4 + 1) * kThreadM + cpj * 4 + 2] += regA[cpi].y * regB[cpj].z;
                    c[(cpi * 4 + 1) * kThreadM + cpj * 4 + 3] += regA[cpi].y * regB[cpj].w;

                    c[(cpi * 4 + 2) * kThreadM + cpj * 4] += regA[cpi].z * regB[cpj].x;
                    c[(cpi * 4 + 2) * kThreadM + cpj * 4 + 1] += regA[cpi].z * regB[cpj].y;
                    c[(cpi * 4 + 2) * kThreadM + cpj * 4 + 2] += regA[cpi].z * regB[cpj].z;
                    c[(cpi * 4 + 2) * kThreadM + cpj * 4 + 3] += regA[cpi].z * regB[cpj].w;

                    c[(cpi * 4 + 3) * kThreadM + cpj * 4] += regA[cpi].w * regB[cpj].x;
                    c[(cpi * 4 + 3) * kThreadM + cpj * 4 + 1] += regA[cpi].w * regB[cpj].y;
                    c[(cpi * 4 + 3) * kThreadM + cpj * 4 + 2] += regA[cpi].w * regB[cpj].z;
                    c[(cpi * 4 + 3) * kThreadM + cpj * 4 + 3] += regA[cpi].w * regB[cpj].w;
                }
            }

            __syncthreads();
        }
    }

    #pragma unroll
    for (int i = 0; i < kThreadM; ++i)
    {
        #pragma unroll
        for (int j = 0; j < kThreadN; j += 4)
        {
            *reinterpret_cast<float4 *>(&regA[0]) = *reinterpret_cast<float4 *>(&baseC[i * dn + j]);
            regA[0].x = regA[0].x * beta + alpha * c[i * kThreadM + j];
            regA[0].y = regA[0].y * beta + alpha * c[i * kThreadM + j + 1];
            regA[0].z = regA[0].z * beta + alpha * c[i * kThreadM + j + 2];
            regA[0].w = regA[0].w * beta + alpha * c[i * kThreadM + j + 3];
            *reinterpret_cast<float4 *>(&baseC[i * dn + j]) = *reinterpret_cast<float4 *>(&regA[0]);
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
    ///                    Disable when debugging output.
    constexpr int kDup = 1;
    constexpr bool kRandInput = true;

    constexpr bool kTestGemmNaive = false;  // Takes too long for m=n=k=4096.
    constexpr bool kTestGemmSmem = true;
    constexpr bool kTestGemmSmemPadTransposedThread = true;
    constexpr bool kTestGemmSmemPadZThread = true;
    constexpr bool kTestGemmWarpTile = true;
    constexpr bool kTestCublasSGemm = true;

    // Problem setting.
    // Tested on NVIDIA Geforce RTX 2080 Ti (kDup=100, kRandInput=true),
    // gemmSmemPad reaches 80% cublasSgemm performance on m=n=k=2048,
    //                     90% cublasSgemm performance on m=n=k=4096.
    // It shows that CUBLAS_DEFAULT_MATH defaults to CUBLAS_TF32_TENSOR_OP_MATH.
    int problemSize = 1024;
    int m = problemSize;
    int n = problemSize;
    int k = problemSize;
    float alpha = 1.0f;
    float beta = 1.0f;
    thrust::host_vector<float> h_a(m * k, 1.0f);
    thrust::host_vector<float> h_b(k * n, 1.0f);
    thrust::host_vector<float> h_c(m * n, 0.0f);

    if constexpr (kRandInput)
    {
        unsigned seed = std::random_device()();
        std::default_random_engine e(seed);
        // std::normal_distribution<float> d(0.0f, 1.0f);
        std::uniform_int_distribution d(1, 20);
        auto g = [&e, &d]() -> float { return d(e); };
        std::generate(h_a.begin(), h_a.end(), g);
        std::generate(h_b.begin(), h_b.end(), g);
        std::generate(h_c.begin(), h_c.end(), g);
    }

    thrust::device_vector<float> golden_c(m * n);
    thrust::device_vector<float> d_a = h_a;
    thrust::device_vector<float> d_b = h_b;
    thrust::device_vector<float> d_c = h_c;

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
    gemmCublas(thrust::raw_pointer_cast(d_a.data()),
               thrust::raw_pointer_cast(d_b.data()),
               thrust::raw_pointer_cast(d_c.data()),
               m,
               n,
               k,
               alpha,
               beta,
               handle);
    golden_c = d_c;

    constexpr int kBlockSize = 16;
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kBlockK = 8;
    constexpr dim3 block(kBlockSize, kBlockSize);
    dim3 grid((n + kBlockN - 1) / kBlockN, (m + kBlockM - 1) / kBlockM);

    // Naive
    if constexpr (kTestGemmNaive)
    {
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
    }

    // Smem
    if constexpr (kTestGemmSmem)
    {
        if constexpr (1 < kDup)
        {
            gemmSmem<kBlockSize, kBlockM, kBlockN, kBlockK><<<grid, block>>>(
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
            gemmSmem<kBlockSize, kBlockM, kBlockN, kBlockK><<<grid, block>>>(
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

        std::printf("gemmSmem: ");
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
    }

    // Smem with Padding
    if constexpr (kTestGemmSmemPadTransposedThread)
    {
        if constexpr (1 < kDup)
        {
            gemmSmemPadTransposedThread<kBlockSize, kBlockM, kBlockN, kBlockK><<<grid, block>>>(
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
            gemmSmemPadTransposedThread<kBlockSize, kBlockM, kBlockN, kBlockK><<<grid, block>>>(
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

        std::printf("gemmSmemPadTransposedThread: ");
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
    }

    // Smem with Padding and z-thread in-warp layout
    if constexpr (kTestGemmSmemPadZThread)
    {
        if constexpr (1 < kDup)
        {
            gemmSmemPadZThread<kBlockSize, kBlockM, kBlockN, kBlockK><<<grid, block>>>(
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
            gemmSmemPadZThread<kBlockSize, kBlockM, kBlockN, kBlockK><<<grid, block>>>(
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

        std::printf("gemmSmemPadZThread: ");
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
    }

    // Smem with Warp Tiling
    if constexpr (kTestGemmWarpTile)
    {
        if constexpr (1 < kDup)
        {
            gemmSmemWarpTile<kBlockSize, kBlockM, kBlockN, kBlockK><<<grid, block>>>(
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
            gemmSmemWarpTile<kBlockSize, kBlockM, kBlockN, kBlockK><<<grid, block>>>(
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

        std::printf("gemmSmemWarpTile: ");
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
    }

    // cublasSgemm.
    if constexpr (kTestCublasSGemm)
    {
        if constexpr (1 < kDup)
        {
            gemmCublas(
                    thrust::raw_pointer_cast(d_a.data()),
                    thrust::raw_pointer_cast(d_b.data()),
                    thrust::raw_pointer_cast(d_c.data()),
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    handle
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        d_c = h_c;
        CUDA_CHECK(cudaEventRecord(ss));

        for (int dup = 0; dup < kDup; ++dup)
        {
            gemmCublas(
                    thrust::raw_pointer_cast(d_a.data()),
                    thrust::raw_pointer_cast(d_b.data()),
                    thrust::raw_pointer_cast(d_c.data()),
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    handle
            );
        }

        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ee));
        CUDA_CHECK(cudaEventSynchronize(ee));

        std::printf("cublasSgemm: ");
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
