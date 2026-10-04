#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
constexpr int THREADS = 256;

void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    std::cerr << expression << ": " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr)

// Teaching version: one block handles the full input.
// Production-sized inputs should use partial sums and a second reduction stage.
__global__ void block_sum_kernel(const float* input, float* output, int n) {
  __shared__ float partial[THREADS];
  const int tid = threadIdx.x;

  float local = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    local += input[i];
  }
  partial[tid] = local;
  __syncthreads();

  // THREADS is a power of two. Every reduction step is block-wide.
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      partial[tid] += partial[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[0] = partial[0];
  }
}
}  // namespace

int main() {
  constexpr int n = 1003;  // tail relative to the block size
  std::vector<float> input(n);
  float reference = 0.0f;
  for (int i = 0; i < n; ++i) {
    input[i] = static_cast<float>((i % 11) - 5);
    reference += input[i];
  }

  float *d_input = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  block_sum_kernel<<<1, THREADS>>>(d_input, d_output, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float actual = 0.0f;
  CUDA_CHECK(cudaMemcpy(&actual, d_output, sizeof(float), cudaMemcpyDeviceToHost));
  if (std::abs(actual - reference) > 1e-3f) {
    std::cerr << "block reduction correctness: FAIL: " << actual
              << " vs " << reference << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "block reduction correctness: PASS\n";

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return EXIT_SUCCESS;
}
