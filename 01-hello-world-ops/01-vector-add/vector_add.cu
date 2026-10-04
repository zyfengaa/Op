#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    std::cerr << expression << ": " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr)

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

}  // namespace

int main() {
  constexpr int n = 100003;  // intentionally not divisible by block size
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> a(n, 1.25f), b(n, -0.5f), c(n, 0.0f);
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

  constexpr int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;
  vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  for (float value : c) {
    if (std::abs(value - 0.75f) > 1e-6f) {
      std::cerr << "vector add correctness: FAIL\n";
      return EXIT_FAILURE;
    }
  }
  std::cout << "vector add correctness: PASS\n";

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return EXIT_SUCCESS;
}
