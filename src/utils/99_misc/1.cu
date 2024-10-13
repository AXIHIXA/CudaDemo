template <typename T, int kWarpThreads = 32>
__device__ T warpReduce(T val)
{
    #pragma unroll
    for (int step = (kWarpThreads >> 1); 0 < step; step >>= 1)
    {
        val += __shfl_xor_sync(0xffffffffu, val, step, kWarpThreads);
    }

    return val;
}


template <int kBlockDimX, typename T, int kWarpThreads = 32>
__global__ void reduce(const T * __restrict__ src,
                       T * __restrict__ dst,
                       int nx)
{
    static_assert(kWarpThreads == 32 && kBlockDimX % kWarpThreads == 0);

    const int tid = threadIdx.x;
    const int bbx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    T x = 0;

    for (int gx = bbx; gx < nx; gx += stride)
    {
        x += src[gx];
    }

    __syncthreads();

    x = warpReduce(x);

    constexpr int kWarps = kBlockDimX / kWarpThreads;
    __shared__ T warpAggregate[kWarps];

    const int warpIdx = tid / kWarpThreads;
    const int laneIdx = tid % kWarpThreads;

    if (laneIdx == 0)
    {
        warpAggregate[warpIdx] = x;
    }

    __syncthreads();

    if (tid == 0)
    {
        x = warpAggregate[0];

        for (int warp = 1; warp < kWarpThreads; ++warp)
        {
            x += warpAggregate[warp];
        }

        dst[blockIdx.x] = x;
    }
}


template <int kBlockDimX, int kBlockDimY, int kPadding = 1, typename T>
__global__ void transpose(const T * __restrict__ src, T * __restrict__ dst, int nx, int ny)
{
    constexpr int kLds = kBlockDimX + kPadding;
    extern __shared__ T smem[];  // (kBlockDimY, kLds)

    int gx = blockIdx.x * kBlockDimX + threadIdx.x;
    int gy = blockIdx.y * kBlockDimY + threadIdx.y;
    int gi = gy * nx + gx;

    int sx = threadIdx.x;
    int sy = threadIdx.y;
    int si = sy * kLds + sx;

    if (gx < nx && gy < ny)
    {
        smem[si] = src[gi];
    }

    __syncthreads();

    const int tid = threadIdx.y * kBlockDimX + threadIdx.x;
    sx = tid / kBlockDimY;
    sy = tid % kBlockDimY;
    si = sy * kLds + sx;

    gx = blockIdx.y * kBlockDimY + sy;
    gy = blockIdx.x * kBlockDimX + sx;
    gi = gy * ny + gx;

    if (gx < ny && gy < nx)
    {
        dst[gi] = smem[si];
    }
}
