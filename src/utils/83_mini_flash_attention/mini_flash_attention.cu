#include <cmath>

#include <cuda_runtime.h>
#include <torch/extension.h>

#include "utils/cuda_utils.h"


template <typename T>
__global__ void flash_attn_fwd_kernel(
        const T * __restrict__ Q,
        const T * __restrict__ K,
        const T * __restrict__ V,
        const int n,
        const int d,
        const int Tc,
        const int Tr,
        const int Bc,
        const int Br,
        const T softmaxScale,
        float * __restrict__ l,
        float * __restrict__ m,
        float * __restrict__ O)
{
    const int tx = threadIdx.x;

    const int batchIdx = blockIdx.x;
    const int headIdx = blockIdx.y;
    const int numHeads = gridDim.y;

    // The base offset is different for each batch and head.
    const int qkvOffset = (batchIdx * numHeads * n * d) + (headIdx * n * d);
    const int lmOffset = (batchIdx * numHeads * n) + (headIdx * n);

    // SMEM for Qi, Kj, Vj, and S == Qi @ Vj.transpose(-2, -1).
    // Qi: (Br, d,)
    // Kj: (Bc, d,)
    // Vj: (Bc, d,)
    // S: (Br, Bc,)
    extern __shared__ float smem[];
    const int tileSize = Bc * d;
    float * Qi = smem;
    float * Kj = Qi + Br * d;
    float * Vj = Kj + Bc * d;
    float * S = Vj + Bc * d;

    for (int j = 0; j < Tc; ++j)
    {
        // Load Kj, Vj from GMEM to SMEM
        for (int x = 0; x < d; ++x)
        {
            // Kj = K[j * Bc:(j + 1) * Bc, :]
            Kj[(tx * d) + x] = K[qkvOffset + (tileSize * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkvOffset + (tileSize * j) + (tx * d) + x];
        }

        __syncthreads();

        for (int i = 0; i < Tr; ++i)
        {
            // Load Qi from GMEM to SMEM
            for (int x = 0; x < d; ++x)
            {
                Qi[(tx * d) + x] = Q[qkvOffset + (tileSize * i) + (tx * d) + x];
            }

            // Reduce rowMax (m) and rowSum (l).
            float oldRowMax = m[lmOffset + (Br * i) + tx];
            float oldRowSum = l[lmOffset + (Br * i) + tx];

            float rowMax = -INFINITY;

            for (int y = 0; y < Bc; ++y)
            {
                float sum = 0;

                for (int x = 0; x < d; ++x)
                {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }

                sum *= softmaxScale;
                S[(Bc * tx) + y] = sum;

                rowMax = max(rowMax, sum);
            }

            float rowSum = 0;

            for (int y = 0; y < Bc; ++y)
            {
                S[(Bc * tx) + y] = expf(S[(Bc * tx) + y] - rowMax);
                rowSum += S[(Bc * tx) + y];
            }

            // Update m and l
            float newRowMax = max(oldRowMax, rowMax);
            float newRowSum = (expf(oldRowMax - newRowMax) * oldRowSum) + (expf(rowMax - newRowMax) * rowSum);

            // Write O, l, m to HBM
            for (int x = 0; x < d; ++x)
            {
                float pv = 0;  // Pij * Vj

                for (int y = 0; y < Bc; ++y)
                {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }

                O[qkvOffset + (tileSize * i) + (tx * d) + x] =
                        (1 / newRowSum) *
                        (
                            (oldRowSum *
                             expf(oldRowMax - newRowMax) *
                             O[qkvOffset + (tileSize * i) + (tx * d) + x])
                            +
                            (expf(rowMax - newRowMax) * pv)
                        );
            }

            m[lmOffset + (Br * i) + tx] = newRowMax;
            l[lmOffset + (Br * i) + tx] = newRowSum;
        }

        __syncthreads();
    }
}


template <typename T>
torch::Tensor fwd(torch::Tensor & Q,
                  torch::Tensor & K,
                  torch::Tensor & V)
{
    const torch::Device device(torch::kCUDA);
    Q = Q.contiguous().to(torch::kFloat32).to(device);
    K = K.contiguous().to(torch::kFloat32).to(device);
    V = V.contiguous().to(torch::kFloat32).to(device);

    constexpr int Bc = 32;
    constexpr int Br = 32;
    static_assert(Bc == Br);

    const int batchSize = Q.size(0);
    const int numHeads = Q.size(1);
    const int seqlen = Q.size(2);
    const int embedDim = Q.size(3);

    const int Tc = (seqlen + Bc - 1) / Bc;
    const int Tr = (seqlen + Br - 1) / Br;

    const T softmaxScale = 1.0 / std::sqrt(embedDim);

    // Initialize O, l, m to HBM
    torch::Tensor O = torch::zeros_like(Q).to(device);
    torch::Tensor l = torch::zeros({batchSize, numHeads, seqlen}).to(torch::kFloat32).to(device);
    torch::Tensor m = torch::full({batchSize, numHeads, seqlen}, -INFINITY).to(torch::kFloat32).to(device);

    // SMEM for Qi, Kj, Vj, and S == Qi @ Vj.T.
    // Qi: (Br, d,)
    // Kj: (Bc, d,)
    // Vj: (Bc, d,)
    // S: (Br, Bc,)
    const int smemBytes = ((Br * embedDim) + (2 * Bc * embedDim) + (Br * Bc)) * sizeof(T);

    // // Note: This returns the max static SMEM size per block.
    // // Per https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability,
    // // NVIDIA Geforce RTX 2080 Ti has 64 KB of SMEM per block (48+ KB requires dynamic allocation).
    // int maxSmemPerBlock;
    // CUDA_CHECK(cudaDeviceGetAttribute(&maxSmemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0));

    if (48 * 1024 < smemBytes)
    {
        CUDA_CHECK(
                cudaFuncSetAttribute(
                        flash_attn_fwd_kernel<T>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        smemBytes
                )
        );
    }

    if (64 * 1024 < smemBytes)
    {
        char buf[1024] = {};
        sprintf(buf,
                "CUDA out of memory. "
                "Max shared memory per block: %d Byte(s); "
                "requested: %d Byte(s)",
                64 * 1024,
                smemBytes);
        throw std::runtime_error(buf);
    }

    dim3 grid(batchSize, numHeads);  // batch_size x num_heads blocks
    dim3 block(Bc);  // Bc threads per block

    flash_attn_fwd_kernel<<<grid, block, smemBytes>>>(
            Q.const_data_ptr<T>(),
            K.const_data_ptr<T>(),
            V.const_data_ptr<T>(),
            seqlen,
            embedDim,
            Tc,
            Tr,
            Bc,
            Br,
            softmaxScale,
            l.mutable_data_ptr<T>(),
            m.mutable_data_ptr<T>(),
            O.mutable_data_ptr<T>()
    );

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    return O;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using T = float;
    using py::literals::operator""_a;

    m.def("fwd",
          torch::wrap_pybind_function(fwd<T>),
          "fwd",
          "Q"_a,
          "K"_a,
          "V"_a);
}
