#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


template <typename T, int kSize>
struct alignas(sizeof(T) * kSize) Vec
{
    T val[kSize];
};


template <typename T>
struct UniformDistribution
{
    __device__ T operator()(curandStatePhilox4_32_10_t * state)
    {
        return static_cast<T>(curand_uniform(state));
    }

    static constexpr int kCount = 1;
};


template <>
struct UniformDistribution<float>
{
    UniformDistribution() = default;

    __device__ float4 operator()(curandStatePhilox4_32_10_t * state)
    {
        return curand_uniform4(state);
    }

    static constexpr int kCount = 4;
};


template <typename T>
struct Dropout
{
    __device__ Dropout(const float dropoutProb,
                       const bool isScale) :
            prob(dropoutProb),
            isScale(isScale),
            invProb(1.0f / (1.0f - dropoutProb))
    {

    }

    __device__ void operator()(const T * __restrict__ src,
                               const T * __restrict__ rand,
                               T * __restrict__ dst)
    {
        static constexpr int kCount = UniformDistribution<T>::kCount;

        #pragma unroll
        for (int i = 0; i < kCount; ++i)
        {
            if (rand[i] < prob)
            {
                const auto zero = static_cast<T>(0);
                dst[i] = zero;
                dst[i + kCount] = zero;
            }
            else
            {
                dst[i] = isScale ? static_cast<T>(src[i] * invProb) : static_cast<T>(src[i]);
                dst[i + kCount] = static_cast<T>(1);
            }
        }
    }

    const float prob;
    const bool isScale;
    float invProb;
};


__global__ void dropoutKernel(int nx,
                              int seed,
                              const float dropoutProb,
                              const float * __restrict__ src,
                              uint8_t * __restrict__ mask,
                              float * __restrict__ dst,
                              bool isScale,
                              int increment,
                              int mainOffset)
{
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    const int bid = blockIdx.x;
    const int blocks = gridDim.x;

    const int blockOffset = bid * threads;
    constexpr int kVecSize = UniformDistribution<float>::kCount;

    // Stride for grid translation, i.e., total number of data handled by the whole grid at one step.
    const int stride = blocks * threads * kVecSize;

    // Initialize cuRAND state.
    // These states are used only once, so they are NOT stored back to GMEM.
    // See "3.5 Performance Notes" available at
    // https://docs.nvidia.com/cuda/curand/device-api-overview.html#performance-notes
    curandStatePhilox4_32_10_t state;
    curand_init(seed, blockOffset + tid, increment, &state);

    UniformDistribution<float> rand4 = {};
    Dropout<float> dropout(dropoutProb, isScale);

    // 0        ~ VecSize - 1     : dst
    // VecSize  ~ 2 * VecSize - 1 : mask
    float regDstMask[kVecSize * 2];
    float regRands[kVecSize];
    uint8_t regMask[kVecSize];

    using fvec4_t = float4;
    using uvec8_t = Vec<uint8_t, kVecSize>;
    fvec4_t temp;

    // Vectorized loads.
    int start = blockOffset * kVecSize;

    for (; start < mainOffset; start += stride)
    {
        // Load
        int threadOffset = tid;
        temp = reinterpret_cast<const fvec4_t *>(src + start)[threadOffset];
        auto r4 = rand4(&state);

        #pragma unroll
        for (int i = 0; i < kVecSize; ++i)
        {
            regDstMask[i] = *(reinterpret_cast<float *>(&temp) + i);
            regRands[i] = static_cast<float>((&r4.x)[i]);
        }

        // Computation
        dropout(&regDstMask[0], &regRands[0], &regDstMask[0]);

        // Write-back
        reinterpret_cast<fvec4_t *>(dst + start)[threadOffset] = *(reinterpret_cast<fvec4_t *>(&regDstMask[0]));

        #pragma unroll
        for (int i = 0; i < kVecSize; i++)
        {
            regMask[i] = static_cast<uint8_t>(regDstMask[i + kVecSize]);
        }

        reinterpret_cast<uvec8_t *>(mask + start)[threadOffset] = *(reinterpret_cast<uvec8_t *>(regMask));
    }

    // Remainder of vectorized loads/stores, use scalar loads/stores.
    int remain = nx - start;

    if (0 < remain)
    {
        // Load
        int threadOffset = tid * kVecSize;
        auto r4 = rand4(&state);

        for (int i = 0; i < kVecSize; i++)
        {
            if (i + threadOffset < remain)
            {
                regDstMask[i] = src[start + threadOffset + i];
            }

            regRands[i] = static_cast<float>((&r4.x)[i]);
        }

        // Computation
        dropout(&regDstMask[0], &regRands[0], &regDstMask[0]);

        // Write-back
        for (int i = 0; i < kVecSize; ++i)
        {
            if ((threadOffset + i) < remain)
            {
                dst[start + threadOffset + i] = regDstMask[i];

                regMask[i] = static_cast<uint8_t>(regDstMask[i + kVecSize]);
                mask[start + threadOffset + i] = regMask[i];
            }
        }
    }
}


int main(int argc, char * argv[])
{
    constexpr bool kTestDropout = true;

    const int nx = 2050;
    thrust::host_vector<float> hostX(nx, 1);
    thrust::host_vector<float> hostY;
    thrust::host_vector<uint8_t> hostMask;

    thrust::device_vector<float> devX = hostX;
    thrust::device_vector<float> devY(nx);
    thrust::device_vector<uint8_t> devMask(nx);

    const bool isScale = true;
    const float dropoutProb = 0.5f;
    const auto seed = static_cast<int>(std::random_device()());
    printf("seed = %u\n", seed);

    if constexpr (kTestDropout)
    {
        constexpr int kRandVecSize = UniformDistribution<float>::kCount;

        dim3 grid(2);
        dim3 block(256);

        // Amount of data that could be loaded/stored vectorized.
        // The remainder are processed as scalar values sequentially.
        // Only applies to 1D data layout.
        const int mainOffset = nx / (grid.x * block.x * kRandVecSize) * (grid.x * block.x * kRandVecSize);

        dropoutKernel<<<grid, block>>>(
                nx,
                seed,
                dropoutProb,
                thrust::raw_pointer_cast(devX.data()),
                thrust::raw_pointer_cast(devMask.data()),
                thrust::raw_pointer_cast(devY.data()),
                isScale,
                0,
                mainOffset
        );

        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());

        hostY = devY;
        hostMask = devMask;

        for (int i = nx - 3; i < nx; ++i)
        {
            std::printf("[%d] y = %f\n", i, hostY[i]);
            std::printf("[%d] mask = %d\n", i, static_cast<int>(hostMask[i]));
        }
    }

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


/*
Compute sanitizer to check memory access:
compute-sanitizer --tool memcheck ./cmake-build-debug/demo
*/