cmake_minimum_required(VERSION 3.24)
project(op_cuda_examples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(vector_add vector_add.cu)
add_executable(naive_gemm naive_gemm.cu)

target_compile_options(vector_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_compile_options(naive_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
