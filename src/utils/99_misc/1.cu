template <int kBlockDimX, int kBlockDimY, int kPadding = 1, typename T>
__global__ void transpose(const T * __restrict__ src, int nx, int ny, T * __restrict__ dst)
{
    // SMEM of shape (kBlockDimY, kLdS). 
    extern __shared__ T smem[];
    constexpr int kLdS = kBlockDimX + kPadding;

    int gx = blockIdx.x * kBlockDimX + threadIdx.x;
    int gy = blockIdx.y * kBlockDimY + threadIdx.y;
    int gi = gy * nx + gx;
    int si = threadIdx.y * kLdS + threadIdx.x;

    if (gx < nx && gy < ny)
    {
        smem[si] = src[gi];
    }

    __syncthreads();

    const int tid = threadIdx.y * kBlockDimX + threadIdx.x;
    const int sx = tid / kBlockDimY;
    const int sy = tid % kBlockDimY;
    si = sy * kLdS + sx;

    gx = blockIdx.y * kBlockDimY + sy;
    gy = blockIdx.x * kBlockDimX + sx;
    gi = gy * ny + gx;
    
    if (gx < ny && gy < nx)
    {
        dst[gi] = smem[si];
    }
}


template <int kBlockDimX, typename T>
__global__ void reduceSmem(const T * __restrict__ src, int nx, T * __restrict__ dst)
{
    constexpr int kWarpThreads = 32;
    static_assert((kBlockDimX & (kBlockDimX - 1)) == 0 && kBlockDimX % kWarpThreads == 0);
    
    __shared__ T smem[kBlockDimX];
    
    const int tid = threadIdx.x;
    const int baseX = blockIdx.x * kBlockDimX + threadIdx.x;
    const int gridStride = gridDim.x * kBlockDimX; 

    smem[tid] = 0;

    for (int gx = baseX; gx < nx; gx += gridStride)
    {
        smem[tid] += src[gx];
    }

    __syncthreads();

    #pragma unroll
    for (int step = (kBlockDimX >> 1); kWarpThreads < step; step >>= 1)
    {
        if (tid < step)
        {
            smem[tid] += smem[tid + step];
        }

        __syncthreads();
    }

    if (tid < kWarpThreads)
    {
        volatile T * vshm = smem;
        T x = vshm[tid];

        if (64 <= kBlockDimX)
        {
            x += vshm[tid + 32]; __syncwarp();
            vshm[tid] = x; __syncwarp();
        }

        x += vshm[tid + 16]; __syncwarp();
        vshm[tid] = x; __syncwarp();
        x += vshm[tid + 8]; __syncwarp();
        vshm[tid] = x; __syncwarp();
        x += vshm[tid + 4]; __syncwarp();
        vshm[tid] = x; __syncwarp();
        x += vshm[tid + 2]; __syncwarp();
        vshm[tid] = x; __syncwarp();
        x += vshm[tid + 1]; __syncwarp();
        vshm[tid] = x; __syncwarp();
    }

    if (tid == 0)
    {
        dst[blockIdx.x] = smem[0];
    }
}


template <typename T>
__device__ T warpReduce(T val)
{
    constexpr unsigned mask = 0xffffffff;
    constexpr int kWarpThreads = 32;

    #pragma unroll
    for (int laneMask = (kWarpThreads >> 1); 0 < laneMask; laneMask >>= 1)
    {
        val += __shfl_xor_sync(mask, val, laneMask, kWarpThreads);
    }
    
    return val;
}


template <int kBlockDimX, typename T>
__global__ void reduceWarp(const T * __restrict__ src, int nx, T * __restrict__ dst)
{
    constexpr int kWarpThreads = 32;
    static_assert((kBlockDimX & (kBlockDimX - 1)) == 0 && kBlockDimX % kWarpThreads == 0);

    constexpr int kWarps = kBlockDimX / kWarpThreads;
    __shared__ T smem[kWarps];
    
    const int tid = threadIdx.x;
    const int baseX = blockIdx.x * kBlockDimX + threadIdx.x;
    const int gridStride = gridDim.x * kBlockDimX; 

    T val = 0;

    for (int gx = baseX; gx < nx; gx += gridStride)
    {
        val += src[gx];
    }

    val = warpReduce(val);

    const int laneIdx = tid % kWarpThreads;
    const int warpIdx = tid / kWarpThreads;

    if (laneIdx == 0)
    {
        smem[warpIdx] = val;
    }

    __syncthreads();

    val = tid < kWarps ? smem[tid] : 0;
    
    if (warpIdx == 0)
    {
        val = warpReduce(val);
    }
    
    if (tid == 0)
    {
        dst[blockDim.x] = val;
    }
}