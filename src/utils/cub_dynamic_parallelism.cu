#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


/// Dynamic parallelism needs nvcc flag --relocatable-device-code=true
/// and set_target_properties(${DEMO} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)


struct Add
{
    __host__ __device__ __forceinline__ float operator()(float a, float b)
    {
        return a + b;
    }
};


struct RowIt
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = float;
    using difference_type = ptrdiff_t;
    using pointer = float *;
    using reference = float &;

    friend __host__ __device__ bool operator!=(const RowIt & a, const RowIt & b);

    friend __host__ __device__ RowIt operator+(const RowIt & a, int diff);

    __host__ __device__ RowIt(float * u, int w, int h) : u(u), w(w), h(h)
    {

    }

    __host__ __device__ float & operator*()
    {
        return *u;
    }

    __host__ __device__ float & operator[](size_t i)
    {
        return u[i * w];
    }

    __host__ __device__ RowIt & operator++()
    {
        u += w;
        return *this;
    }

    float * u;
    int w;
    int h;
};


__host__ __device__ bool operator!=(const RowIt & a, const RowIt & b)
{
    return a.u != b.u;
}


__host__ __device__ RowIt operator+(const RowIt & a, int diff)
{
    return {a.u + a.w * diff, a.w, a.h};
}


namespace std
{

template <>
struct iterator_traits<RowIt>
{
    using iterator_category = RowIt::iterator_category;
    using value_type = RowIt::value_type;
    using difference_type = RowIt::difference_type;
    using pointer = RowIt::pointer;
    using reference = RowIt::reference;
};

}  // namespace std


__global__ void scan1(float * u, int ww, int hh)
{
//    extern __shared__ unsigned char smem[];
//    float * su = reinterpret_cast<float *>(smem);

    size_t tempStorageBytes = 0;
    Add add;
    cub::DeviceScan::InclusiveScan(nullptr, tempStorageBytes, u, u, add, ww);
    float * tempStorage = nullptr;
    cudaMalloc(&tempStorage, tempStorageBytes * hh);

    for (int h = 0; h < hh; ++h)
    {
        cub::DeviceScan::InclusiveScan(tempStorage + h * ww, tempStorageBytes, u + h * ww, u + h * ww, add, ww);
    }

    for (int w = 0; w < ww; ++w)
    {
        cub::DeviceScan::InclusiveScan(tempStorage + w * hh, tempStorageBytes, RowIt(u + w, ww, hh), RowIt(u + w, ww, hh), add, ww);
    }

    __syncthreads();
    cudaFree(tempStorage);
}


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

    unsigned tIdxInBlock = threadIdx.y * blockDim.x + threadIdx.x;

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
}


__global__ void scan2(float * u, float * v, int ww, int hh)
{
    size_t tempStorageBytes = 0;
    Add add;
    cub::DeviceScan::InclusiveScan(nullptr, tempStorageBytes, u, u, add, ww);
    float * tempStorage = nullptr;
    cudaMalloc(&tempStorage, tempStorageBytes * hh);

    for (int h = 0; h < hh; ++h)
    {
        cub::DeviceScan::InclusiveScan(tempStorage + h * ww, tempStorageBytes, u + h * ww, u + h * ww, add, ww);
    }

    constexpr dim3 block = {32, 32};
    dim3 grid = {(ww + block.x - 1) / block.x, (hh + block.y - 1) / block.y};
    transpose<<<grid, block, (block.x + 1) * block.y * sizeof(float)>>>(u, ww, hh, v);
    __syncthreads();

    // TODO: w != h
    for (int h = 0; h < hh; ++h)
    {
        cub::DeviceScan::InclusiveScan(tempStorage + h * ww, tempStorageBytes, v + h * ww, v + h * ww, add, ww);
    }

    transpose<<<grid, block, (block.x + 1) * block.y * sizeof(float)>>>(v, ww, hh, u);

    __syncthreads();
    cudaFree(tempStorage);
}


int main(int argc, char * argv[])
{
    constexpr int r = 256;
    constexpr int c = 256;
    constexpr int rc = r * c;
    thrust::host_vector<float> h_in(rc);

    std::random_device rd;
    unsigned long long seed = rd();
    std::default_random_engine e(seed);
    std::normal_distribution<float> nd;
    auto gen = [&nd, &e]() { return nd(e); };

    thrust::generate(thrust::host, h_in.begin(), h_in.end(), gen);

    thrust::device_vector<float> d_in1 = h_in;

    using Clock = std::chrono::high_resolution_clock;

    auto start = Clock::now();
    for (int dup = 0; dup < 100; ++dup)
    {
        scan1<<<1, 1>>>(d_in1.data().get(), c, r);
    }
    cudaDeviceSynchronize();
    auto end = Clock::now();
    printf("scan1 x1000 took %ld ms.\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    thrust::host_vector<float> h_out1 = d_in1;

    thrust::device_vector<float> d_in2 = h_in;

    start = Clock::now();
    for (int dup = 0; dup < 100; ++dup)
    {
        thrust::device_vector<float> d_temp(rc);
        scan2<<<1, 1>>>(d_in2.data().get(), d_temp.data().get(), c, r);
    }
    cudaDeviceSynchronize();
    end = Clock::now();
    printf("scan2 x1000 took %ld ms.\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    thrust::host_vector<float> h_out2 = d_in2;

//    for (int i = 0; i < r * c; ++i)
//    {
//        if (1e-6f < std::abs(h_out1[i] - h_out2[i]))
//        {
//            printf("WRONG!\n");
//            break;
//        }
//    }
//
//    for (int i = 0; i < r; ++i)
//    {
//        for (int j = 0; j < c; ++j)
//        {
//            std::printf("%+4.2f ", h_out1[i * c + j]);
//        }
//
//        std::printf("\n");
//    }
//
//    std::printf("\n");
//
//    for (int i = 0; i < r; ++i)
//    {
//        for (int j = 0; j < c; ++j)
//        {
//            std::printf("%+4.2f ", h_out2[i * c + j]);
//        }
//
//        std::printf("\n");
//    }
//
//    std::printf("\n");

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