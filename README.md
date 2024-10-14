# CudaDemo

Personal CUDA demo repository. 
Implemented multiple CUDA kernels and profiles different optimization routines. 
More details are available in `src/utils/`. 

## Implemented Kernels (And Benchmarked Variants)

- Reduction: `src/utils/05_reduce`
  - Naive GEMM
  - SMEM version
  - Multiple elements per thread
  - Loop unrolling and last-warp unrolled
  - Warp shuffle reduction
  - Grid-translation to adapt arbitraty sizes
- Histogram: `src/utils/07_hist`
  - SMEM histogram program with atomic primitives
- Copy-If (Filter): `src/utils/08_copy_if`
  - Naive GEMM
  - Tiled SMEM, in-block atomicAdd
  - Warp atomic aggregation with inlined PTX (`%lanemask_lt`)
- Fused Biased-Mask-Scale-Add: `src/utils/10_fused_biased_mask_scale_add`
  - FP32 and FP16 versions
  - For FP16: Vectorized FP16 arithmetics (`__hadd2`, etc.)
- Softmax: `src/utils/11_softmax`
  - Implemented at warp's perspective (Registers directly, no SMEM).
  - Vectorized loads and stores as optmization. 
- Measure Peak Performance: `src/utils/12_measure_peak_performance`
  - Evaluates GMEM bandwidth by vectorized loads. 
    - GMEM bandwidth is evaluated by oversized loads (so that L2 cache gets flushed.)
  - Evaluates peak computing performance by FP32 FMAs while saturating all available SMs. 
- CUDA Streams: `src/utils/13_cuda_streams`
  - Experiments with cudaMemcpyAsync and non-default-stream kernel launches. 
    - Must use pinned host memory. 
    - Overlaps kernel execution with host-device memory transfer. 
  - Observed performance boost with non-default streams.
- GEMV: `src/utils/15_gemv`
  - GEMV kernel for fp32.
  - GEVM kernel for both fp32 and fp16.
  - Uses vectorized loads and SMEM optimization.
- Dropout: `src/utils/16_dropout`
  - Fuse mask generation and scaling into one kernel.
- Matrix Transpose: `src/utils/80_transpose`
  - Naive GMEM
  - Padded SMEM Version with no bank conflicts.
- Im2col: `src/utils/81_im2col`
  - Im2col kernel for convolution routines.
- Parallel Scan: `src/utils/82_scan`
  - Block scan routine with warp shuffle intrinsics. 
  - Supports thread-level unrollment (each thread could handle multiple elements).
- SGEMM: `src/utils/89_gemm`
  - Reaches 90% performance (avg over 100 times) of cuBLAS on 4096x4096x4096 FP32 GEMM
  - Naive GEMM
  - Naive tiled SMEM (with bank conflicts) with vectorized loads and stores
    - Bank conflicts take place at stores and loads;
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
