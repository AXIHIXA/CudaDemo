#define TEST_REDUCE_V5
//#define TEST_REDUCE_V6

#ifdef TEST_REDUCE_V5
#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define THREAD_PER_BLOCK 256
// latency: 0.656ms
// v5：循环展开
template <int blockSize>
__device__ void BlockSharedMemReduce(float* smem) {
    //对v4 L45的for循环展开，以减去for循环中的加法指令，以及给编译器更多重排指令的空间
  if (blockSize >= 1024) {
    if (threadIdx.x < 512) {
      smem[threadIdx.x] += smem[threadIdx.x + 512];
    }
    __syncthreads();
  }
  if (blockSize >= 512) {
    if (threadIdx.x < 256) {
      smem[threadIdx.x] += smem[threadIdx.x + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (threadIdx.x < 128) {
      smem[threadIdx.x] += smem[threadIdx.x + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (threadIdx.x < 64) {
      smem[threadIdx.x] += smem[threadIdx.x + 64];
    }
    __syncthreads();
  }
  // the final warp
  if (threadIdx.x < 32) {
    volatile float* vshm = smem;
    if (blockDim.x >= 64) {
      vshm[threadIdx.x] += vshm[threadIdx.x + 32];
    }
    vshm[threadIdx.x] += vshm[threadIdx.x + 16];
    vshm[threadIdx.x] += vshm[threadIdx.x + 8];
    vshm[threadIdx.x] += vshm[threadIdx.x + 4];
    vshm[threadIdx.x] += vshm[threadIdx.x + 2];
    vshm[threadIdx.x] += vshm[threadIdx.x + 1];
  }
}

template <int blockSize>
__global__ void reduce_v5(float *d_in, float *d_out){
    __shared__ float smem[THREAD_PER_BLOCK];
    // 泛指当前线程在其block内的id
    unsigned int tid = threadIdx.x;
    // 泛指当前线程在所有block范围内的全局id, *2代表当前block要处理2*blocksize的数据
    // ep. blocksize = 2, blockIdx.x = 1, when tid = 0, gtid = 4, gtid + blockSize = 6; when tid = 1, gtid = 5, gtid + blockSize = 7
    // ep. blocksize = 2, blockIdx.x = 0, when tid = 0, gtid = 0, gtid + blockSize = 2; when tid = 1, gtid = 1, gtid + blockSize = 3
    // so, we can understand L59, one thread handle data located in tid and tid + blockSize
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // load: 每个线程加载两个元素到shared mem对应位置
    smem[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();
    // compute: reduce in shared mem
    BlockSharedMemReduce<blockSize>(smem);

    // store: 哪里来回哪里去，把reduce结果写回显存
    // GridSize个block内部的reduce sum已得出，保存到d_out的每个索引位置
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

bool CheckResult(float *out, float groudtruth, int n){
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    if (res != groudtruth) {
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;

    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    //int GridSize = 100000;
    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float*)malloc((GridSize) * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(float));

    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize / 2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i)
    reduce_v5<blockSize / 2><<<Grid,Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d \n", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < GridSize;i++){
            printf("resPerBlock : %lf ",out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_v5 latency = %f ms\n", milliseconds / 100.0f);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
#endif  // TEST_REDUCE_V5

#ifdef TEST_REDUCE_V6
#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
// 注意: v0-v5里面kernel得到的是各个block负责范围内的总和，要想得到最终的和，需要把各个block求得的总和再做reduce sum
// v6: multi-block reduce final result by two pass
// latency: 1.815ms
template <int blockSize>
__device__ void BlockSharedMemReduce(float* smem) {
    //对v4 L45的for循环展开，以减去for循环中的加法指令，以及给编译器更多重排指令的空间
  if (blockSize >= 1024) {
    if (threadIdx.x < 512) {
      smem[threadIdx.x] += smem[threadIdx.x + 512];
    }
    __syncthreads();
  }
  if (blockSize >= 512) {
    if (threadIdx.x < 256) {
      smem[threadIdx.x] += smem[threadIdx.x + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (threadIdx.x < 128) {
      smem[threadIdx.x] += smem[threadIdx.x + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (threadIdx.x < 64) {
      smem[threadIdx.x] += smem[threadIdx.x + 64];
    }
    __syncthreads();
  }
  // the final warp
  if (threadIdx.x < 32) {
    volatile float* vshm = smem;
    if (blockDim.x >= 64) {
      vshm[threadIdx.x] += vshm[threadIdx.x + 32];
    }
    vshm[threadIdx.x] += vshm[threadIdx.x + 16];
    vshm[threadIdx.x] += vshm[threadIdx.x + 8];
    vshm[threadIdx.x] += vshm[threadIdx.x + 4];
    vshm[threadIdx.x] += vshm[threadIdx.x + 2];                                                                                                                                                                                          vshm[threadIdx.x] += vshm[threadIdx.x + 1];

  }
}

template <int blockSize>
__global__ void reduce_v6(float *d_in, float *d_out, int nums){
    __shared__ float smem[blockSize];
    // 泛指当前线程在其block内的id
    unsigned int tid = threadIdx.x;
    // 泛指当前线程在所有block范围内的全局id
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_thread_num = blockDim.x * gridDim.x;
    // 基于v5的改进：不用显式指定一个线程处理2个元素，而是通过L58的for循环来自动确定每个线程处理的元素个数
    float sum = 0.0f;
    for (int32_t i = gtid; i < nums; i += total_thread_num) {
        sum += d_in[i];
    }
    smem[tid] = sum;
    __syncthreads();
    // compute: reduce in shared mem
    BlockSharedMemReduce<blockSize>(smem);

    // store: 哪里来回哪里去，把reduce结果写回显存
    // GridSize个block内部的reduce sum已得出，保存到d_out的每个索引位置
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

bool CheckResult(float *out, float groudtruth, int n){
    if (*out != groudtruth) {
      return false;
    }
    return true;
}

int main(){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxblocks = deviceProp.maxGridSize[0];
    const int blockSize = 256;
    const int N = 25600000;
    int gridSize = std::min((N + blockSize - 1) / blockSize, maxblocks);

    float milliseconds = 0;
    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a,N * sizeof(float));

    float *out = (float*)malloc((gridSize) * sizeof(float));
    float *d_out;
    float *part_out;//新增part_out存储每个block reduce的结果
    cudaMalloc((void **)&d_out, 1 * sizeof(float));
    cudaMalloc((void **)&part_out, (gridSize) * sizeof(float));
    float groudtruth = N;

    for(int i = 0; i < N; i++){
        a[i] = 1;
    }

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(gridSize);
    dim3 Block(blockSize);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
    reduce_v6<blockSize><<<Grid, Block>>>(d_a, part_out, N);
    reduce_v6<blockSize><<<1, Block>>>(part_out, d_out, gridSize); }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(out, groudtruth, 1);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0;i < 1;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }
    printf("reduce_v6 latency = %f ms\n", milliseconds / 100.0f);

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(part_out);
    free(a);
    free(out);
}
#endif  // TEST_REDUCE_V6
