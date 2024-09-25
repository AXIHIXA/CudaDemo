#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"


template <typename T, int kSize>
struct alignas(sizeof(T) * kSize) Vec
{
    __host__ __device__ inline const T & operator[](int i) const
    {
        return val[i];
    }

    __host__ __device__ inline T & operator[](int i)
    {
        return const_cast<T &>((static_cast<const Vec &>(*this))[i]);
    }

    T val[kSize];
};


__device__ float ftanh(const float x)
{
    float y;
    asm("{ tanh.approx.f32 %0,%1; }\n" : "=f"(y) : "f"(x));
    return y;
}


__device__ void hgelu2(const half * __restrict__ in, half * __restrict__ out)
{
    const half2 x = *(reinterpret_cast<const half2 *>(in));
    const half2 cube_x = __hmul2(__hmul2(x, x), x);
    const half2 x_plus_beta_cube_x = __hadd2(x, __hmul2(__float2half2_rn(0.044714998453855515f), cube_x));
    const half2 tanh_arg = __hmul2(__float2half2_rn(0.7978845608028654f), x_plus_beta_cube_x);
    half2 tanh_res;
    tanh_res.x = __float2half(ftanh(__half2float(tanh_arg.x)));
    tanh_res.y = __float2half(ftanh(__half2float(tanh_arg.y)));
    const half2 y = __hmul2(__hmul2(__float2half2_rn(0.5f), x),
                            __hadd2(__float2half2_rn(1.0f), tanh_res));
    *reinterpret_cast<half2 *>(out) = y;
    *reinterpret_cast<half2 *>(out) = x;
}


/// 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044714998453855515 * x^3)))
float fgelu(float x)
{
    return 0.5f * x * std::tanh(0.7978845608028654f * (x + 0.044714998453855515f * x * x * x));
}


template <int kVecSize, int kBlockSize>
__global__ void gelu(const half * __restrict__ in, int nx, half * __restrict__ out)
{
    using Vec = Vec<half, kVecSize>;

    int offset = (blockIdx.x * kBlockSize + threadIdx.x) * kVecSize;
    const int stride = gridDim.x * kBlockSize * kVecSize;

    half y[kVecSize];

    for (; offset < nx; offset += stride)
    {
        const half * __restrict__ x = in + offset;

        for (int i = 0; i < kVecSize; i += 2)
        {
            hgelu2(x + i, y + i);
        }

        *reinterpret_cast<Vec *>(out + offset) = *reinterpret_cast<Vec *>(y);
    }
}


int main(int argc, char * argv[])
{
    constexpr int n = 1'000;
    thrust::host_vector<half> h_x(n);
    thrust::host_vector<half> h_y(n);

    // Using iota with int32 will break internal structure of h_x!!!
    for (int i = 0; i < n; ++i)
    {
        h_x[i] = static_cast<half>(i);
    }

    thrust::device_vector<half> d_x = h_x;
    thrust::device_vector<half> d_y(n);

    constexpr dim3 block = {512};
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp deviceProp = {};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    constexpr int kVecSize = 8;
    dim3 grid = {std::min<unsigned>((n / kVecSize + block.x - 1) / block.x, deviceProp.maxGridSize[0])};
    gelu<kVecSize, block.x><<<grid, block>>>(d_x.data().get(), n, d_y.data().get());
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    h_y = d_y;

    for (int i = 0; i < n; ++i)
    {
        std::printf("%f vs %f\n", __half2float(h_y[i]), fgelu(h_x[i]));
    }

    return EXIT_SUCCESS;
}

