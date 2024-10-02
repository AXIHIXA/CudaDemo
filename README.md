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
- Matrix Transpose
  - Naive GMEM
  - Padded SMEM Version with no bank conflicts
- SGEMM: Reaches 90% performance (avg over 100 times) of cuBLAS on 4096x4096x4096 FP32 SGEMM
  - Naive GEMM
  - Naive tiled SMEM (with bank conflicts) with vectorized loads and stores
  - Padded SMEM with wrap tiling (no bank conflict)
  - Pure warp tiling variants (no bank conflict), transposed tiling and z-tiling
  - Double-buffer optimization
