#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


struct Add
{
    __device__ __forceinline__ float operator()(float a, float b)
    {
        return a + b;
    }
};


template <int kBlockDim, int kNItems>
__global__ void testBlockLoadScanStore(const float * __restrict__ in, int n, float * __restrict__ out)
{
    // cub::BLOCK_LOAD_WARP_TRANSPOSE:
    // Suppose the input buffer is in blocked arrangement [[1 2 3 4] [5 6 7 8] [9 10 11 12] [13 14 15 16]],
    // where thread #0's data is supposed to own [1 2 3 4], #1 own [5 6 7 8], ...
    // Threads read input in a striped manner, i.e., #0 reads 1 5 9 13, #1 reads 2 6 10 14, ...
    // So the reads in a warp are coalesced,
    // and the temp storage is like [[1 5 9 13] [2 6 10 14] [3 7 11 15] [4 8 12 16]].
    // The temp storage is then locally transposed into [[1 2 3 4] [5 6 7 8] [9 10 11 12] [13 14 15 16]].
    // (It's much slower to transpose gmem than to transpose smem.)
    // See https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockLoad.html
    using BlockLoad = cub::BlockLoad<float, kBlockDim, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockScan = cub::BlockScan<float, kBlockDim>;

    // Same, first transpose local thread data, then write in striped manner (which is coalesced at warp-scale).
    // See https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockStore.html
    using BlockStore = cub::BlockStore<float, kBlockDim, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    extern __shared__ unsigned char smem[];
    auto & loadTempStorage = reinterpret_cast<typename BlockLoad::TempStorage &>(smem);
    auto & scanTempStorage = reinterpret_cast<typename BlockScan::TempStorage &>(smem);
    auto & storeTempStorage = reinterpret_cast<typename BlockStore::TempStorage &>(smem);

    float threadData[kNItems];

    BlockLoad(loadTempStorage).Load(in + blockIdx.x * kNItems, threadData);
    __syncwarp();  // Must invoke to sequre smem safety.

    BlockScan(scanTempStorage).InclusiveScan(threadData, threadData, Add());
    __syncwarp();  // Must invoke to sequre smem safety.

    BlockStore(storeTempStorage).Store(out, threadData);
    __syncwarp();  // Must invoke to sequre smem safety.
}


int main(int argc, char * argv[])
{
    constexpr int n = 1024;
    thrust::host_vector<float> h_in(n, 1.0f);
    thrust::device_vector<float> d_in = h_in;
    thrust::device_vector<float> d_out(n, 0.0f);

    constexpr int kBlockDim = 128;
    constexpr int kNItems = 4;

    testBlockLoadScanStore<kBlockDim, kNItems>
            <<<1, kBlockDim, sizeof(float) * kBlockDim * kNItems>>>(
            d_in.data().get(), n, d_out.data().get());
    CUDA_CHECK_LAST_ERROR();
    thrust::host_vector<float> h_out = d_out;

    for (int i = 0; i < kBlockDim * kNItems; ++i)
    {
        std::printf("%f\n", h_out[i]);
    }

    return EXIT_SUCCESS;
}
