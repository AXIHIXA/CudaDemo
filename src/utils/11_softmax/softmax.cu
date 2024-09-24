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


int main(int argc, char * argv[])
{
    constexpr int nx = 1'024;
    constexpr int ny = 1'000;
    constexpr int n = nx * ny;
    constexpr int kDup = 1;

    thrust::host_vector<float> h_x(n, 1.0f);
    thrust::host_vector<float> h_y(n);
    thrust::host_vector<float> gt(n);
    cpuSoftmax(h_x.data(), nx, ny, gt.data());

    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y(n);

    constexpr dim3 block = {32, 8};
    constexpr dim3 grid = {1, 125};

    float ms;
    cudaEvent_t ss, ee;
    CUDA_CHECK(cudaEventCreate(&ss));
    CUDA_CHECK(cudaEventCreate(&ee));

    // Test.
    thrust::fill(d_y.begin(), d_y.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int i = 0; i < kDup; ++i)
    {
        softmax<1, nx / kWarpSize, kWarpSize, 1><<<grid, block>>>(
            d_x.data().get(),
            d_y.data().get(),
            nx,
            ny);
    }

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    h_y = d_y;
    std::printf("softmax took %f ms, ", ms / kDup);
    checkResult(h_y.data(), gt.data(), nx, ny);


    // Free cuda events.
    cudaEventDestroy(ss);
    cudaEventDestroy(ee);

    return EXIT_SUCCESS;
}
