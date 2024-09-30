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


void histCpu(const int * __restrict__ in, int nx, int * __restrict__ out)
{
    for (int i = 0; i < nx; ++i)
    {
        ++out[in[i]];
    }
}


template <int kBlockSize, int kMaxVal>
__global__ void histSmem(const int * __restrict__ in, int nx, int * __restrict__ out)
{
    extern __shared__ int cnt[];

    const int tid = threadIdx.x;
    const int bx = blockIdx.x * kBlockSize;
    const int numThreadsInGrid = gridDim.x * kBlockSize;

    if (tid < kMaxVal)
    {
        cnt[tid] = 0;
    }

    __syncthreads();

    for (int i = bx + threadIdx.x; i < nx; i += numThreadsInGrid)
    {
        atomicAdd(cnt + in[i], 1);
    }

    __syncthreads();

    if (tid == 0)
    {
        for (int i = 0; i < kMaxVal; ++i)
        {
            atomicAdd(out + i, cnt[i]);
        }
    }
}


void checkResult(const thrust::host_vector<int> & res,
                 const thrust::host_vector<int> & gt)
{
    std::printf("Result is %s\n\n", res == gt ? "correct." : "WRONG!!!");

    printf("res: ");
    for (int x : res)
    {
        printf("%d ", x);
    }
    printf("\ngt : ");
    for (int x : gt)
    {
        printf("%d ", x);
    }
    printf("\n\n");
}


int main(int argc, char * argv[])
{
    constexpr int nIn = 25'600'000;
    constexpr int nOut = 256;
    thrust::host_vector<int> h_in(nIn);
    thrust::host_vector<int> h_out;

    for (int i = 0; i < nIn; ++i)
    {
        h_in[i] = i % nOut;
    }

    thrust::host_vector<int> gt(nOut, 0);
    histCpu(h_in.data(), nIn, gt.data());

    thrust::device_vector<int> d_in = h_in;
    thrust::device_vector<int> d_out(nOut, 0);

    constexpr dim3 block = {256};
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp deviceProp = {};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    dim3 grid = {std::min<unsigned>((nIn + block.x - 1) / block.x, deviceProp.maxGridSize[0])};

    float ms;
    cudaEvent_t ss;
    cudaEvent_t ee;
    CUDA_CHECK(cudaEventCreate(&ss));
    CUDA_CHECK(cudaEventCreate(&ee));

    CUDA_CHECK(cudaEventRecord(ss));
    histSmem<block.x, nOut><<<grid, block, nOut * sizeof(int)>>>(
            thrust::raw_pointer_cast(d_in.data()),
            nIn,
            thrust::raw_pointer_cast(d_out.data())
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ee));
    CUDA_CHECK(cudaEventSynchronize(ee));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ss, ee));

    h_out = d_out;
    std::printf("histSmem took %f ms, ", ms);
    checkResult(h_out, gt);

    CUDA_CHECK(cudaEventDestroy(ss));
    CUDA_CHECK(cudaEventDestroy(ee));

    return EXIT_SUCCESS;
}
