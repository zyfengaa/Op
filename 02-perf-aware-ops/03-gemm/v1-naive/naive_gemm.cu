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

__global__ void naive_gemm_kernel(const float* a, const float* b, float* c,
                                  int m, int n, int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }
  float acc = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    acc += a[row * k + inner] * b[inner * n + col];
  }
  c[row * n + col] = acc;
}

std::vector<float> cpu_gemm(const std::vector<float>& a,
                            const std::vector<float>& b, int m, int n, int k) {
  std::vector<float> c(static_cast<size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      for (int inner = 0; inner < k; ++inner) {
        c[row * n + col] += a[row * k + inner] * b[inner * n + col];
      }
    }
  }
  return c;
}

}  // namespace

int main() {
  constexpr int m = 37;
  constexpr int n = 29;
  constexpr int k = 41;
  const size_t a_bytes = static_cast<size_t>(m) * k * sizeof(float);
  const size_t b_bytes = static_cast<size_t>(k) * n * sizeof(float);
  const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
  std::vector<float> a(m * k), b(k * n), c(m * n);
  for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
  for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<float>(static_cast<int>(i % 7) - 3);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
  CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
  CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), a_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), b_bytes, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  naive_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

  const auto expected = cpu_gemm(a, b, m, n, k);
  for (size_t i = 0; i < c.size(); ++i) {
    if (!std::isfinite(c[i]) || std::abs(c[i] - expected[i]) > 1e-4f) {
      std::cerr << "naive GEMM correctness: FAIL at " << i << '\n';
      return EXIT_FAILURE;
    }
  }
  std::cout << "naive GEMM correctness: PASS\n";

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return EXIT_SUCCESS;
}
