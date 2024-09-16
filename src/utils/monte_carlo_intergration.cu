#include <random>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include "util/CudaUtil.h"


/// Monte-Carlo intergration routine inside unit circle.
/// Estimates value of PI.
__global__
void iint(
        const float2 * __restrict__ sample,
        int len,
        int * __restrict__ mask
)
{
    auto idx = static_cast<int>(blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x);

    if (idx < len)
    {
        float2 r = sample[idx];
        mask[idx] = r.x * r.x + r.y * r.y <= 1.0f;
    }
}


class UniformFloat2
{
public:
    UniformFloat2() = delete;

    UniformFloat2(unsigned int seed, float xMin, float xMax, float yMin, float yMax)
            : e(seed), dx(xMin, xMax), dy(yMin, yMax)
    {
        // Nothing needed here.
    }

    __host__ __device__
    float2 operator()(unsigned long long i)
    {
        e.discard(i);
        return {dx(e), dy(e)};
    }

private:
    thrust::default_random_engine e;
    thrust::uniform_real_distribution<float> dx;
    thrust::uniform_real_distribution<float> dy;
};


int main(int argc, char * argv[])
{
    static constexpr int kNumSamples {50000000};
    static constexpr dim3 kBlockDim {32U, 32U, 1U};
    static constexpr unsigned int kBlockSize {kBlockDim.x * kBlockDim.y * kBlockDim.z};

    unsigned int seed = std::random_device()();
    std::printf("seed = %u\n", seed);

    int numSamples = kNumSamples;
    thrust::device_vector<float2> dSample(numSamples);

    thrust::transform(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(numSamples),
        dSample.begin(),
        UniformFloat2(seed, -1.0f, 1.0f, -1.0f, 1.0f)
    );

    thrust::device_vector<int> dMask(numSamples, 0);

    dim3 iintGridDim {(numSamples + kBlockSize - 1U) / kBlockSize, 1U, 1U};

    iint<<<iintGridDim, kBlockDim, 0U, nullptr>>>(
            dSample.data().get(),
            numSamples,
            dMask.data().get()
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    int numInside = thrust::reduce(thrust::device, dMask.begin(), dMask.end());
    std::printf("Monte-Carlo PI = %lf\n", static_cast<double>(numInside) / static_cast<double>(numSamples) * 4.0);

    return EXIT_SUCCESS;
}
