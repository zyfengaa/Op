#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
constexpr int TILE = 16;

void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    std::cerr << expression << ": " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr)

// One block computes a TILE x TILE output tile.
// A/B values are cooperatively loaded into shared memory and reused.
__global__ void tiled_gemm_kernel(const float* a, const float* b, float* c,
                                  int m, int n, int k) {
  __shared__ float tile_a[TILE][TILE];
  __shared__ float tile_b[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  const int k_tiles = (k + TILE - 1) / TILE;

  for (int tile = 0; tile < k_tiles; ++tile) {
    const int a_col = tile * TILE + threadIdx.x;
    const int b_row = tile * TILE + threadIdx.y;

    tile_a[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

    __syncthreads();

    #pragma unroll
    for (int inner = 0; inner < TILE; ++inner) {
      acc += tile_a[threadIdx.y][inner] * tile_b[inner][threadIdx.x];
    }

    // Do not overwrite a tile while another thread still consumes it.
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = acc;
  }
}

std::vector<float> cpu_gemm(const std::vector<float>& a,
                            const std::vector<float>& b, int m, int n, int k) {
  std::vector<float> c(static_cast<size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < n; ++col)
      for (int inner = 0; inner < k; ++inner)
        c[row * n + col] += a[row * k + inner] * b[inner * n + col];
  return c;
}
}  // namespace

int main() {
  // All three dimensions have tails relative to TILE.
  constexpr int m = 37, n = 29, k = 41;
  const size_t a_bytes = static_cast<size_t>(m) * k * sizeof(float);
  const size_t b_bytes = static_cast<size_t>(k) * n * sizeof(float);
  const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
  std::vector<float> a(m * k), b(k * n), c(m * n);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<float>(static_cast<int>(i % 7) - 3);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
  CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
  CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), a_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), b_bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  tiled_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

  const auto expected = cpu_gemm(a, b, m, n, k);
  for (size_t i = 0; i < c.size(); ++i) {
    if (!std::isfinite(c[i]) || std::abs(c[i] - expected[i]) > 1e-4f) {
      std::cerr << "tiled GEMM correctness: FAIL at " << i << '\n';
      return EXIT_FAILURE;
    }
  }
  std::cout << "tiled GEMM correctness: PASS\n";

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return EXIT_SUCCESS;
}
