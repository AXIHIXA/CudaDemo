#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


template <typename T>
__global__ void transposeNaive(const T * __restrict__ in, int nx, int ny, T * __restrict__ out)
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned from = iy * nx + ix;
    unsigned to = ix * ny + iy;

    if (ix < nx && iy < ny)
    {
        out[to] = in[from];
    }
}


/// Read input in row-major, store into smem (padded) in row-major.
/// Write to output in row major.
/// This will essentially read smem in column-major.
/// That is, a thread will NOT write the data it reads.
///
/// Do NOT exit if thread id is out of bound!
/// Because threads do not write what they read,
/// margin (remainder) elements will be read by in-bound threads
/// (with these in-bound threads write NOTHING)
/// but written by out-of-bound threads!
template <int kPadding = 1, typename T>
__global__ void transpose(const T * __restrict__ in, int nx, int ny, T * __restrict__ out)
{
    extern __shared__ T smem[];

    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned from = iy * nx + ix;
    unsigned s = threadIdx.y * (blockDim.x + kPadding) + threadIdx.x;

    if (ix < nx && iy < ny)
    {
        smem[s] = in[from];
    }

    __syncthreads();

    unsigned t = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned sx = t / blockDim.y;
    unsigned sy = t % blockDim.y;
    s = sy * (blockDim.x + kPadding) + sx;

    ix = blockIdx.y * blockDim.y + sy;
    iy = blockIdx.x * blockDim.x + sx;
    unsigned to = iy * ny + ix;

    if (ix < ny && iy < nx)
    {
        out[to] = smem[s];
    }
}


void print(const thrust::host_vector<float> & in, int r, int c, const thrust::host_vector<float> & out)
{
    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            printf("%.1f ", in[i * c + j]);
        }

        std::printf("\n");
    }

    std::printf("\n");

    for (int i = 0; i < c; ++i)
    {
        for (int j = 0; j < r; ++j)
        {
            printf("%.1f ", out[i * r + j]);
        }

        std::printf("\n");
    }

    std::printf("\n");
}


void checkResult(const thrust::host_vector<float> & in, int r, int c, const thrust::host_vector<float> & out)
{
    #ifndef NDEBUG
    print(in, r, c, out);
    #endif

    bool resultIsCorrect = true;

    for (int i = 0, shouldBreak = false; !shouldBreak && i < r; ++i)
    {
        for (int j = 0; !shouldBreak && j < c; ++j)
        {
            if (in[i * c + j] != out[j * r + i])
            {
                resultIsCorrect = false;
                shouldBreak = true;
            }
        }
    }

    std::printf("Result is %s\n\n", resultIsCorrect ? "correct." : "WRONG!!!");
}


int main(int argc, char * argv[])
{
    constexpr int r = 2051;
    constexpr int c = 4089;
    constexpr int rc = r * c;
    thrust::host_vector<float> h_in(rc);
    std::iota(h_in.begin(), h_in.end(), 0.0f);
    thrust::device_vector<float> d_in = h_in;
    thrust::device_vector<float> d_out(rc);
    thrust::host_vector<float> h_out;

    #ifdef NDEBUG
    constexpr int kDup = 100;
    #else
    constexpr int kDup = 1;
    #endif  // NDEBUG

    constexpr int kPadding = 1;
    constexpr dim3 block = {32, 32};
    dim3 grid = {(c + block.x - 1) / block.x, (r + block.y - 1) / block.y};

    float ms;
    cudaEvent_t ss;
    cudaEvent_t ee;
    CUDA_CHECK(cudaEventCreate(&ss));
    CUDA_CHECK(cudaEventCreate(&ee));

    // Naive
    #ifdef NDEBUG
    transposeNaive<<<grid, block>>>(d_in.data().get(), c, r, d_out.data().get());
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    #endif  // NDEBUG

    thrust::fill(d_out.begin(), d_out.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int dup = 0; dup < kDup; ++dup)
    {
        transposeNaive<<<grid, block>>>(d_in.data().get(), c, r, d_out.data().get());
    }

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));

    h_out = d_out;
    std::printf("transposeNaive: ");
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    std::printf("took %f ms, ", ms / kDup);
    checkResult(h_in, r, c, h_out);

    // Regular
    #ifdef NDEBUG
    transpose<kPadding><<<grid, block, (block.x + kPadding) * block.y * sizeof(float)>>>(
        d_in.data().get(), c, r, d_out.data().get());
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    #endif  // NDEBUG

    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0.0f);
    CUDA_CHECK(cudaEventRecord(ss));

    for (int dup = 0; dup < kDup; ++dup)
    {
        transpose<kPadding><<<grid, block, (block.x + kPadding) * block.y * sizeof(float)>>>(
            d_in.data().get(), c, r, d_out.data().get());
    }

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));

    h_out = d_out;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));
    std::printf("took %f ms, ", ms / kDup);
    checkResult(h_in, r, c, h_out);

    CUDA_CHECK(cudaEventDestroy(ss));
    CUDA_CHECK(cudaEventDestroy(ee));

    return EXIT_SUCCESS;
}

/*
# Profile gld_throughput, gld_efficiency, gst_throughput and gst_efficiency.
ncu -k regex:transpose --metrics \
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \
./cmake-build-release/demo

# Profile all common metrics.
ncu -k regex:transpose ./cmake-build-release/demo

# For runtime profiling.
nvprof ./cmake-build-release/demo
*/