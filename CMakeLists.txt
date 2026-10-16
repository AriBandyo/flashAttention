cmake_minimum_required(VERSION 3.25)
project(FlashForge LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

add_library(flashforge_cuda STATIC
  cuda/src/v0_naive.cu
  cuda/src/v1_online.cu
  cuda/src/v2_fa2_partition.cu
  cuda/src/decode_splitkv.cu)
target_include_directories(flashforge_cuda PUBLIC cuda/include)
target_compile_options(flashforge_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>)

add_executable(flashforge_bench cuda/src/bench_cuda.cu)
target_link_libraries(flashforge_bench PRIVATE flashforge_cuda)
target_include_directories(flashforge_bench PRIVATE cuda/include)

add_executable(flashforge_test cuda/src/test_cuda.cu)
target_link_libraries(flashforge_test PRIVATE flashforge_cuda)
target_include_directories(flashforge_test PRIVATE cuda/include)
