cmake_minimum_required(VERSION 3.24)
project(op_cuda_examples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(vector_add ../../01-hello-world-ops/01-vector-add/vector_add.cu)
add_executable(block_reduce_sum ../../01-hello-world-ops/03-reduction-sum-max/block_reduce_sum.cu)
add_executable(naive_gemm ../../02-perf-aware-ops/03-gemm/v1-naive/naive_gemm.cu)
add_executable(tiled_gemm ../../02-perf-aware-ops/03-gemm/v2-tiled/tiled_gemm.cu)

target_compile_options(vector_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_compile_options(block_reduce_sum PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_compile_options(naive_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_compile_options(tiled_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
