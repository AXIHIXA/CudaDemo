cmake_minimum_required(VERSION 3.25)
project(CudaDemo CUDA CXX)

# Ensure correct CUDA architecture
# NVIDIA Geforce RTX 2080 Ti has Compute Capability 7.5
# https://developer.nvidia.com/cuda-gpus
# https://stackoverflow.com/questions/67794606/cmake-cuda-architecture-flags
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(ALL_INCLUDE_DIRS
        include
)

set(ALL_LIBRARIES
        cudadevrt
        cudart
        cublas
        curand
        cusparse
)

set(ALL_COMPILE_DEFS

)

set(X_PTXAS_V
        -Xptxas -v
)

set(ALL_COMPILE_OPTS
        # -g is default in CMAKE_<LANG>_FLAGS_DEBUG
        $<$<CONFIG:DEBUG>:-O0>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:${X_PTXAS_V}>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code=true>
)

set(DEMO demo)
add_executable(${DEMO}
        include/utils/cuda_utils.h
        src/main.cu
)
set_target_properties(${DEMO} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_definitions(${DEMO} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${DEMO} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${DEMO} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${DEMO} ${ALL_LIBRARIES})

# NOT NEEDED given CMAKE_CUDA_ARCHITECTURE is set properly
# target_compile_options(${EXECUTABLE} PRIVATE
#         $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_75,code=[compute_75,sm_75]>)
