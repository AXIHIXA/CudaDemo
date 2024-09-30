#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


static constexpr int kWarpSize = 32;


/// Softmax on innermost dimension.
void cpuSoftmax(const float * __restrict__ in, int nx, int ny, float * __restrict__ out)
{
    for (int j = 0; j < ny; ++j)
    {
        float sigma = 0.0f;
        float maxi = 0.0f;

        for (int i = 0; i < nx; ++i)
        {
            maxi = std::max(in[j * nx + i], maxi);
        }

        for (int i = 0; i < nx; ++i)
        {
            sigma += std::exp(in[j * nx + i] - maxi);
        }

        for (int i = 0; i < nx; ++i)
        {
            out[j * nx + i] = std::exp(in[j * nx + i] - maxi) / sigma;
        }
    }
}


template <typename T, int kVecSize>
struct alignas(sizeof(T) * kVecSize) Vec
{
    T val[kVecSize];
};

template <typename T>
struct Sum
{
    __device__ __forceinline__ T operator()(const T & a, const T & b) const
    {
        return a + b;
    }
};

template <typename T>
struct Max
{
    __device__ __forceinline__ T operator()(const T & a, const T & b) const
    {
        return max(a, b);
    }
};


template <template <typename> class ReductionOp, typename T, int kWarpWidth = kWarpSize>
__inline__ __device__ T warpReduce(T val)
{
    #pragma unroll
    for (int mask = kWarpWidth >> 1; 0 < mask; mask >>= 1)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }

    return val;
}


template<int pack_size, int cols_per_thread,
         int warp_width, int rows_per_thread>
__global__ void softmax(const float* src, float* dst, const int cols, const int rows) {
  constexpr int num_packs = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * warp_width);
  float buf[rows_per_thread][cols_per_thread];
  //当前warp在所有warp中的id号，因为每行表示一个warp，所以只需求得列号，即global warp id
  const int global_warp_id = blockIdx.y * blockDim.y + threadIdx.y;
  const int num_global_warp = gridDim.y * blockDim.y; // 125 * 8 = 1000, 与src.rows()匹配
  const int lane_id = threadIdx.x;
  const int step = num_global_warp * rows_per_thread; // 1000
  // 进入到当前所分配的整个block数量的数值处理范围
  for (int row = global_warp_id * rows_per_thread; row < rows; row += step) {
    float thread_max[rows_per_thread];
    // 细粒度化，进入到每个线程所处理的行数范围
    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
      thread_max[row_id] = -Inf<float>();
      float* row_buf = buf[row_id];
      // 再细粒度一点，进入到每个线程所处理的一行的多个向量范围
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        // 每个向量的起始偏移
        const int pack_offset = pack_id * pack_size;
        // 当前向量所在的起始列号
        const int col = (pack_id * warp_width + lane_id) * pack_size;
        if (col < cols) {
          // 根据起始列号，读取当前向量到row_buf寄存器
          load<pack_size>(src, row_buf + pack_offset, row + row_id, cols, col);
          // 求出pack  local和thread local的最大值
          for (int i = 0; i < pack_size; ++i) {
            thread_max[row_id] = max(thread_max[row_id], row_buf[pack_offset + i]);
          }
        } else {
          // 起始列号超出了总列数，则设为负无穷，对softmax值无影响
          for (int i = 0; i < pack_size; ++i) { row_buf[pack_offset + i] = -Inf<float>(); }
        }
      }
    }
    // 声明rows_per_thread个寄存器保存当前线程计算的行的最大值
    float warp_max[rows_per_thread];
    // reduce各个线程计算的最大值，得出所有线程中的最大值，即一行的最大值
    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
      warp_max[row_id] = WarpReduce<MaxOp, float, warp_width>(thread_max[row_id]);
    }
    // 声明rows_per_thread个寄存器保存当前线程计算的行的总和，即softmax分母
    float thread_sum[rows_per_thread];

    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
      thread_sum[row_id] = 0;
      float* row_buf = buf[row_id];
      // 当前线程拥有的row_buf值的总和，softmax分母的partial value
      for (int i = 0; i < cols_per_thread; ++i) {
        row_buf[i] = Exp(row_buf[i] - warp_max[row_id]);
        thread_sum[row_id] += row_buf[i];
      }
    }
    float warp_sum[rows_per_thread];
    // softmax分母的final value
    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
      warp_sum[row_id] = WarpReduce<SumOp, float, warp_width>(thread_sum[row_id]);
    }

    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
      float* row_buf = buf[row_id];
      // 分子除分母得到sfotmax最终结果
      for (int i = 0; i < cols_per_thread; ++i) {
        row_buf[i] = Div(row_buf[i], warp_sum[row_id]);
      }
      // 哪里来回哪里去，把最终结果写回显存
      for (int i = 0; i < num_packs; ++i) {
        const int col = (i * warp_width + lane_id) * pack_size;
        if (col < cols) {
          store<pack_size>(dst, row_buf + i * pack_size, row + row_id, cols, col);
        }
      }
    }
  }
}



void checkResult(const float * __restrict__ pred, const float * __restrict__ gt, int nx, int ny)
{
    bool correct = true;

    for (int i = 0; i < nx * ny; ++i)
    {
        if (pred[i] != gt[i])
        {
            correct = false;
            break;
        }
    }

    std::printf("result is %s\n", correct ? "correct." : "WRONG!!!");

    #if false
    for (int i = 0; i < nx * ny; ++i)
    {
        printf("%f %f\n", pred[i], gt[i]);
    }
    #endif
}


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
    constexpr bool kTestGemmSmemWarpTile = true;
    constexpr bool kTestGemmSmemDoubleBuffer = true;
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
    if constexpr (kTestGemmSmemWarpTile)
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

    // Double buffer.
    if constexpr (kTestGemmSmemDoubleBuffer)
    {
        if constexpr (1 < kDup)
        {
            gemmSmemDoubleBuffer<kBlockSize, kBlockM, kBlockN, kBlockK><<<grid, block>>>(
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
            gemmSmemDoubleBuffer<kBlockSize, kBlockM, kBlockN, kBlockK><<<grid, block>>>(
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

        std::printf("gemmSmemDoubleBuffer: ");
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
