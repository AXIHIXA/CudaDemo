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


__global__ void transposeNaive(const float * __restrict__ in, int nx, int ny, float * __restrict__ out)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < nx && y < ny)
    {
        unsigned from = y * nx + x;
        unsigned to = x * ny + y;
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
template <int kPadding = 1>
__global__ void transpose(const float * __restrict__ in, int nx, int ny, float * __restrict__ out)
{
    extern __shared__ float smem[];  // smem[blockDim.y][blockDim.x + kPadding].

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

    // This thread will write t-th (column major) element in smem.
    // What's its x, y coordinates in this block?
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


/// "Row unroll":
/// Each thread block is "duplicated" for kNItems times along the x-dimension.
/// Threads first do what they do in the "original block",
/// then they move to the next block and do what they do in that area.
///
/// E.g.
/// [[ 0  1  2  3  4  5  6  7],
///  [ 8  9 10 11 12 13 14 15]].
/// blockDim == {2, 2}, kNItems == 2.
/// Threads in block #0 first load [0, 1, 8, 9] into smem,
/// then load [2, 3, 9, 10].
/// For sequential loads in for loop, each thread simply offset x coordinate by blockDim.x.
/// Smem index also offsets by blockDim.x. (The padding does not affect here as we don't cross rows.)
template <int kPadding = 1, int kNItems = 1>
__global__ void transposeUnroll(const float * __restrict__ in, int nx, int ny, float * __restrict__ out)
{
    extern __shared__ float smem[];

    unsigned ix = blockIdx.x * (blockDim.x * kNItems) + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned from = iy * nx + ix;
    unsigned s = threadIdx.y * (blockDim.x * kNItems + kPadding) + threadIdx.x;

    #pragma unroll
    for (int u = 0; u < kNItems && ix + blockDim.x * u < nx && iy < ny; ++u)
    {
        smem[s + blockDim.x * u] = in[from + blockDim.x * u];

        #ifndef NDEBUG

        if (blockIdx.x == 0 && blockIdx.y == 0)
        {
            printf("t%u%u smem[%u] <- in[%u] (%f)\n",
                   threadIdx.y, threadIdx.x,
                   s + blockDim.x * u, from + blockDim.x * u, in[from + blockDim.x * u]);
        }
        #endif  // NDEBUG
    }

    __syncthreads();

    // t is the INDEX of this thread in block, not index of its first element!
    unsigned t = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned sx = t / blockDim.y;
    unsigned sy = t % blockDim.y;
    s = sy * (blockDim.x * kNItems + kPadding) + sx;

    ix = blockIdx.y * blockDim.y + sy;
    iy = blockIdx.x * (blockDim.x * kNItems) + sx;
    unsigned to = iy * ny + ix;

    #pragma unroll
    for (int u = 0; u < kNItems && ix < ny && iy + blockDim.x * u < nx; ++u)
    {
        // The next element-to-write for this thread spans blockDim.x * blockDim.y elements (row-major)
        // in the output block (of x size blockDim.y, y size blockDim.x), i.e., blockDim.x rows.
        // The row-major index in output array spans by blockDim.x * ny.
        out[to + ny * blockDim.x * u] = smem[s + blockDim.x * u];

        #ifndef NDEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0)
        {
            printf("t%u%u out[%u] <- smem[%u] (%f)\n",
                   threadIdx.y, threadIdx.x,
                   to + ny * blockDim.x * u, s + blockDim.x * u, smem[s + blockDim.x * u]);
        }
        #endif  // NDEBUG
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
    constexpr int r = 4;
    constexpr int c = 8;
    constexpr int rc = r * c;
    thrust::host_vector<float> h_in(rc);
    std::iota(h_in.begin(), h_in.end(), 0.0f);
    thrust::device_vector<float> d_in = h_in;
    thrust::device_vector<float> d_out(rc);
    thrust::host_vector<float> h_out;

    constexpr dim3 block = {2, 2};
    dim3 grid = {(c + block.x - 1) / block.x, (r + block.y - 1) / block.y};

    thrust::fill(d_out.begin(), d_out.end(), 0.0f);
    transposeNaive<<<grid, block>>>(d_in.data().get(), c, r, d_out.data().get());
    CUDA_CHECK(cudaDeviceSynchronize());
    h_out = d_out;
    std::printf("transposeNaive: ");
    checkResult(h_in, r, c, h_out);

    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0.0f);
    constexpr int kPad = 1;
    transpose<<<grid, block, (block.x + kPad) * block.y * sizeof(float)>>>(
            d_in.data().get(), c, r, d_out.data().get());
    CUDA_CHECK(cudaDeviceSynchronize());
    h_out = d_out;
    std::printf("transpose: ");
    checkResult(h_in, r, c, h_out);

    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0.0f);
    constexpr int kNItems = 2;
    transposeUnroll<kPad, kNItems><<<grid, block, (block.x * kNItems + kPad) * block.y * sizeof(float)>>>(
            d_in.data().get(), c, r, d_out.data().get());
    CUDA_CHECK(cudaDeviceSynchronize());
    h_out = d_out;
    std::printf("transposeUnroll: ");
    checkResult(h_in, r, c, h_out);

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