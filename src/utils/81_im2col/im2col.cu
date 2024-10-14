#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


// See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
// Input tensor im of shape (inputChannels, height, width),
// convolution kernel of shape (kernelHeight, kernelWidth),
// padded with (padHeight, padWidth),
// with stride (strideHeight, strideWidth) and dilation (dilationHeight, dilationWidth).
// Output tensor col of shape (colChannels, colHeight, colWidth).
// with colChannels == inputChannels * kernelHeight * kernelWidth,
// and colHeight, colWidth as specified in document above.
// Multiple batches are handled by sequential kernel launches (on the same stream).
template <typename T>
__global__ void im2colKernel(
        const int n,
        const T * __restrict__ imPtr,
        const int height,
        const int width,
        const int kernelHeight,
        const int kernelWidth,
        const int padHeight,
        const int padWidth,
        const int strideHeight,
        const int strideWidth,
        const int dilationHeight,
        const int dilationWidth,
        const int colHeight,
        const int colWidth,
        T * __restrict__ colPtr)
{
    // Grid translation.
    // Actually, grid translation will NOT happen when we launch a sufficient number of blocks.
    // n == outputChannels == channels * colHeight * colWidth.
    // Each thread copies kernelHeight * kernelWidth elements from imPtr to colPtr.
    // The output colPtr points to tensor of size: (inputChannels * kernelHeight * kernelWidth,
    //                                              colHeight,
    //                                              colWidth)
    for (int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < n; gid += gridDim.x * blockDim.x)
    {
        const int wOut = gid % colWidth;

        const int idx = gid / colWidth;

        const int hOut = idx % colHeight;
        const int cIn = idx / colHeight;
        const int cOut = cIn * kernelHeight * kernelWidth;

        const int hIn = hOut * strideHeight - padHeight;
        const int wIn = wOut * strideWidth - padWidth;

        T * col = colPtr + (cOut * colHeight + hOut) * colWidth + wOut;
        const T * im = imPtr + (cIn * height + hIn) * width + wIn;

        for (int i = 0; i < kernelHeight; ++i)
        {
            for (int j = 0; j < kernelWidth; ++j)
            {
                const int h = hIn + i * dilationHeight;
                const int w = wIn + j * dilationWidth;

                *col = (0 <= h && h < height && 0 <= w && w < width) ?
                       im[i * dilationHeight * width + j * dilationWidth] :
                       static_cast<T>(0);

                col += colHeight * colWidth;
            }
        }
    }
}


template <typename T>
void im2col(
        cudaStream_t stream,
        const T * __restrict__ imPtr,
        const int channels,
        const int height,
        const int width,
        const int kernelHeight,
        const int kernelWidth,
        const int padHeight,
        const int padWidth,
        const int strideHeight,
        const int strideWidth,
        const int dilationHeight,
        const int dilationWidth,
        const int colHeight,
        const int colWidth,
        T * __restrict__ colPtr)
{
    // We will launch channels * colHeight * colWidth threads,
    // each thread will be responsible for copying a single-channel grid covered by the kernel,
    // i.e., kernelHeight * kernelWidth elements.
    static constexpr int kThreadsPerBlock = 1024;
    const int threads = channels * height * width;
    dim3 block((threads + kThreadsPerBlock - 1) / kThreadsPerBlock);

    im2colKernel<<<block, kThreadsPerBlock, 0, stream>>>(
            threads,
            imPtr,
            height,
            width,
            kernelHeight,
            kernelWidth,
            padHeight,
            padWidth,
            strideHeight,
            strideWidth,
            dilationHeight,
            dilationWidth,
            colHeight,
            colWidth,
            colPtr
    );

    CUDA_CHECK_LAST_ERROR();
}


int main(int argc, char * argv[])
{
    using T = float;

    const int batchSize = 1;
    const int channels = 3;
    const int height = 3;
    const int width = 3;
    const int kernelHeight = 2;
    const int kernelWidth = 2;
    const int padHeight = 0;
    const int padWidth = 0;
    const int strideHeight = 1;
    const int strideWidth = 1;
    const int dilationHeight = 1;
    const int dilationWidth = 1;

    const int colChannels = channels * kernelHeight * kernelWidth;
    const int colHeight = (height + 2 * padHeight - (dilationHeight * (kernelHeight - 1) + 1)) / strideHeight + 1;
    const int colWidth = (width + 2 * padWidth - (dilationWidth * (kernelWidth - 1) + 1)) / strideWidth + 1;

    thrust::host_vector<T> hstIm(batchSize * channels * height * width);
    std::iota(hstIm.begin(), hstIm.end(), 1);

    thrust::device_vector<T> devIm = hstIm;
    thrust::device_vector<T> devCol(batchSize * colChannels * colHeight * colWidth, 233);

    T * imPtr = thrust::raw_pointer_cast(devIm.data());
    T * colPtr = thrust::raw_pointer_cast(devCol.data());

    // CUDA resources that require manual destruction.
    float ms;
    cudaEvent_t ss, ee;
    CUDA_CHECK(cudaEventCreate(&ss));
    CUDA_CHECK(cudaEventCreate(&ee));

    im2col(nullptr,
           imPtr,
           channels,
           height,
           width,
           kernelHeight,
           kernelWidth,
           padHeight,
           padWidth,
           strideHeight,
           strideWidth,
           dilationHeight,
           dilationWidth,
           colHeight,
           colWidth,
           colPtr);

    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::host_vector<T> hstCol = devCol;
    printf("col:\n");

    for (int y = 0; y < colChannels; ++y)
    {
        for (int x = 0; x < colHeight * colWidth; ++x)
        {
            printf("%6.2f ", hstCol[y * colHeight * colWidth + x]);
        }

        printf("\n");
    }

    printf("\n");

    // Free cuda resources.
    CUDA_CHECK(cudaEventDestroy(ss));
    CUDA_CHECK(cudaEventDestroy(ee));

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
