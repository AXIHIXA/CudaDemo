#include <cassert>
#include <cmath>
#include <iostream>

#include <cuda_runtime.h>

#include "utils/cuda_utils.h"


template <typename T>
__global__ void add(T * x, T * y, T * z, int nx)
{
    int gx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = gx; i < nx; i += gridDim.x * blockDim.x)
    {
        z[i] = y[i] + x[i];
    }
}

template <typename T>
void addCpu(T * x, T * y, T * z, int nx)
{
    for (int i = 0; i < nx; i++)
    {
        z[i] = y[i] + x[i];
    }
}


int main(int argc, char * argv[])
{
    using T = float;

    // Default stream took 1.933568 ms
    // Async stream took 1.511424 ms
    const int arraySize = 1000000;
    const int numBytes = arraySize * sizeof(T);
    const int numStreams = 4;
    const bool forceDefaultStream = true;

    assert(arraySize % numStreams == 0);
    const int nx = arraySize / numStreams;
    const int nxBytes = nx * sizeof(T);

    T * hx;
    T * hy;
    T * hz;

    // MUST manually alllocate pinned memory using cudaHostAlloc
    CUDA_CHECK(cudaHostAlloc(&hx, numBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hy, numBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hz, numBytes, cudaHostAllocDefault));

    for (int i = 0; i < arraySize; ++i)
    {
        hx[i] = 1.0f;
        hy[i] = 1.0f;
    }

    auto hz_cpu = reinterpret_cast<T *>(malloc(numBytes));
    addCpu(hx, hy, hz_cpu, arraySize);

    T * dx;
    T * dy;
    T * dz;

    CUDA_CHECK(cudaMalloc(&dx, numBytes));
    CUDA_CHECK(cudaMalloc(&dy, numBytes));
    CUDA_CHECK(cudaMalloc(&dz, numBytes));

    cudaStream_t streams[numStreams];

    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    const dim3 block(256);
    dim3 grid((nxBytes + block.x - 1) / block.x);

    float ms;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < numStreams; ++i)
    {
        int offset = i * nx;

        cudaMemcpyAsync(dx + offset,
                        hx + offset,
                        nxBytes,
                        cudaMemcpyHostToDevice,
                        forceDefaultStream ? nullptr : streams[i]);

        cudaMemcpyAsync(dy + offset,
                        hy + offset,
                        nxBytes,
                        cudaMemcpyHostToDevice,
                        forceDefaultStream ? nullptr : streams[i]);

        add<<<grid, block, 0, forceDefaultStream ? nullptr : streams[i]>>>(
                dx + offset,
                dy + offset,
                dz + offset,
                nx);

        cudaMemcpyAsync(hz + offset,
                        dz + offset,
                        nxBytes, cudaMemcpyDeviceToHost,
                        forceDefaultStream ? nullptr : streams[i]);
    }

    // Use cudaDeviceSynchronize to sync host and all streams of device.
    // When we only need sync one stream and device,
    // use cudaStreamSynchronize, which is light-weight than multiple cudaStreamSynchronize() s.
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("%s stream took %f ms\n", forceDefaultStream ? "Default" : "Async", ms);

    for (int i = 0; i < arraySize; ++i)
    {
        if (1e-6f < fabs(hz_cpu[i] - hz[i]))
        {
            printf("index: %d, cpu: %f, gpu: %f\n", i, hz_cpu[i], hz[i]);
            break;
        }
    }

    printf("Result right\n");

    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }

    CUDA_CHECK(cudaFreeHost(hx));
    CUDA_CHECK(cudaFreeHost(hy));
    CUDA_CHECK(cudaFreeHost(hz));

    free(hz_cpu);

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dz));

    return 0;
}