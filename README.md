# CudaDemo

Personal CUDA demo repository. 
Implemented multiple CUDA kernels and profiles different optimization routines. 
More details are available in `src/utils/`. 

## Implemented Kernels (And Benchmarked Variants)

- Reduction
  - Naive GEMM
  - SMEM version
  - In-block unroll (multiple elements per block)
  - Full loop unroll
  - warp shuffle reduction
  - Grid-translation to adapt arbitraty sizes
- Histogram
  - SMEM histogram program with atomic primitives
- Copy-If (Filter)
  - Naive GEMM
  - Tiled SMEM, in-block atomicAdd
  - Warp atomic aggregation with inlined PTX (`%lanemask_lt`)
- Fused Biased-Mask-Scale-Add
  - FP32 and FP16 versions
  - For FP16: Vectorized FP16 arithmetics (`__hadd2`, etc.)
- Softmax
  - Implemented at warp's perspective (Registers directly, no SMEM).
  - Vectorized loads and stores as optmization. 
- Measure Peak Performance
  - Evaluates GMEM bandwidth by vectorized loads. 
    - GMEM bandwidth is evaluated by oversized loads (so that L2 cache gets flushed.)
  - Evaluates peak computing performance by FP32 FMAs while saturating all available SMs. 
- CUDA Streams
  - Experiments with cudaMemcpyAsync and non-default-stream kernel launches. 
    - Must use pinned host memory. 
  - Observed performance boost with non-default streams. 
- Matrix Transpose
  - Naive GMEM
  - Padded SMEM Version with no bank conflicts
- SGEMM: Reaches 90% performance (avg over 100 times) of cuBLAS on 4096x4096x4096 FP32 SGEMM
  - Naive GEMM
  - Naive tiled SMEM (with bank conflicts) with vectorized loads and stores
    - Bank conflicts take place at stores and loads. 
      - Each thread block handles 128x128 block;
      - Each thread handles one 8x8 block or 4 strided 4x4 blocks (depending on tiling);
      - Each SMEM chunk contains 8x128 elements. 
    - Stores:
      - Padding or warp tiling.
    - Loads: 
      - Each thread vectorized-loads two float4 s, so each phase contains 8 threads. 
      - Tile threads in warp s.t. no consecutive threads-of-8s conflicts at the same bank. 
      - Could access the same bank address (but not same bank with different addresses!)
  - Padded SMEM with wrap tiling (no bank conflict)
  - Pure warp tiling variants (no bank conflict), transposed tiling and z-tiling
  - Double-buffer optimization
