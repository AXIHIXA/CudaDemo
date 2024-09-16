#include <algorithm>
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


template <int kPadding = 1>
__global__ void transpose(const float * __restrict__ in, int nx, int ny, float * __restrict__ out)
{
    extern __shared__ float smem[];  // smem[blockDim.y][blockDim.x + kPadding].

    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;

    // Read input in row-major, store into smem (padded) in row-major.
    // Write to output in row major.
    // This will essentially read smem in column-major.
    // That is, a thread will NOT write the data it reads.

    // Do NOT exit if thread id is out of bound!
    // Because threads do not write what they read,
    // margin (remainder) elements will be read by in-bound threads
    // (with these in-bound threads write NOTHING)
    // but written by out-of-bound threads!

    unsigned from = iy * nx + ix;
    unsigned s = threadIdx.y * (blockDim.x + kPadding) + threadIdx.x;

    if (ix < nx && iy < ny)
    {
        smem[s] = in[from];
    }

    __syncthreads();

    unsigned tIdxInBlock = threadIdx.y * blockDim.x + threadIdx.x;

    #ifndef NDEBUG
    if (blockIdx.y == -1)
    {
        printf("b[%u] t[%u] -- smem[%u] = in[%u] (%f)\n",
               blockIdx.y * blockDim.x + blockIdx.x, tIdxInBlock,
               s, from, smem[s]);
    }
    #endif

    unsigned sx = tIdxInBlock / blockDim.y;
    unsigned sy = tIdxInBlock % blockDim.y;
    s = sy * (blockDim.x + kPadding) + sx;

    ix = blockIdx.y * blockDim.y + sy;
    iy = blockIdx.x * blockDim.x + sx;
    unsigned to = iy * ny + ix;

    if (ix < ny && iy < nx)
    {
        out[to] = smem[s];
    }

    #ifndef NDEBUG
    if (blockIdx.y == -1)
    {
        printf("b[%u] t[%u] -- out[%u] = smem[%u] (%f)\n",
               blockIdx.y * blockDim.x + blockIdx.x, tIdxInBlock,
               to, s, smem[s]);
    }
    #endif
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

    #ifndef NDEBUG
    print(in, r, c, out);
    #endif
}


int main(int argc, char * argv[])
{
    constexpr int r = 2997;
    constexpr int c = 1993;
    constexpr int rc = r * c;
    thrust::host_vector<float> h_in(rc);
    std::iota(h_in.begin(), h_in.end(), 0.0f);
    thrust::device_vector<float> d_in = h_in;
    thrust::device_vector<float> d_out(rc);
    thrust::host_vector<float> h_out;

    constexpr dim3 block = {32, 32};
    dim3 grid = {(c + block.x - 1) / block.x, (r + block.y - 1) / block.y};

    thrust::fill(d_out.begin(), d_out.end(), 0.0f);
    transposeNaive<<<grid, block>>>(d_in.data().get(), c, r, d_out.data().get());
    CUDA_CHECK(cudaDeviceSynchronize());
    h_out = d_out;
    checkResult(h_in, r, c, h_out);

    thrust::fill(thrust::device, d_out.begin(), d_out.end(), 0.0f);
    constexpr int kPad = 1;
    transpose<kPad><<<grid, block, (block.x + kPad) * block.y * sizeof(float)>>>(
            d_in.data().get(), c, r, d_out.data().get());
    CUDA_CHECK(cudaDeviceSynchronize());
    h_out = d_out;
    checkResult(h_in, r, c, h_out);

    return EXIT_SUCCESS;
}

/*
ncu -k regex:transpose --metrics \
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \
./cmake-build-release/demo
*/