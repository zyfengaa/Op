// Teaching implementation: contiguous FP32 rows; finite inputs; forward only.
// Run without arguments for correctness, or --benchmark for a device-time sample.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

constexpr int THREADS = 256;  // Power of two, required by the reduction below.

void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    std::cerr << expression << ": " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr)

// Every thread must call this helper. It ends with a barrier AFTER reading s[0]:
// without it, a fast thread could overwrite s[0] during the next reduction.
template <bool MAXIMUM>
__device__ float block_reduce(float value, float* s) {
  const int t = threadIdx.x;
  s[t] = value;
  __syncthreads();
  for (int step = blockDim.x / 2; step > 0; step /= 2) {
    if (t < step) {
      s[t] = MAXIMUM ? fmaxf(s[t], s[t + step]) : s[t] + s[t + step];
    }
    __syncthreads();
  }
  const float result = s[0];
  __syncthreads();
  return result;
}

__global__ void row_softmax(const float* x, float* y, int width) {
  __shared__ float scratch[THREADS];
  const int t = threadIdx.x;
  const size_t base = static_cast<size_t>(blockIdx.x) * width;

  // Stage 1: each thread scans columns t, t+256, t+512, ...
  float local_max = -CUDART_INF_F;
  for (int col = t; col < width; col += blockDim.x) {
    local_max = fmaxf(local_max, x[base + col]);
  }
  const float m = block_reduce<true>(local_max, scratch);

  // Stage 2: aggregate shifted exponentials; inactive threads contribute zero.
  float local_sum = 0.0f;
  for (int col = t; col < width; col += blockDim.x) {
    local_sum += expf(x[base + col] - m);
  }
  const float denominator = block_reduce<false>(local_sum, scratch);

  // Stage 3: re-read/recompute for simplicity instead of retaining a whole row.
  for (int col = t; col < width; col += blockDim.x) {
    y[base + col] = expf(x[base + col] - m) / denominator;
  }
}

__global__ void row_layernorm(const float* x, const float* gamma,
                              const float* beta, float* y, int width, float eps) {
  __shared__ float scratch[THREADS];
  const int t = threadIdx.x;
  const size_t base = static_cast<size_t>(blockIdx.x) * width;

  float local_sum = 0.0f;
  for (int col = t; col < width; col += blockDim.x) {
    local_sum += x[base + col];
  }
  const float mean = block_reduce<false>(local_sum, scratch) / width;

  float local_m2 = 0.0f;
  for (int col = t; col < width; col += blockDim.x) {
    const float diff = x[base + col] - mean;
    local_m2 += diff * diff;
  }
  const float variance = block_reduce<false>(local_m2, scratch) / width;
  const float inv_std = rsqrtf(variance + eps);
  for (int col = t; col < width; col += blockDim.x) {
    y[base + col] = (x[base + col] - mean) * inv_std * gamma[col] + beta[col];
  }
}

std::vector<double> reference(const std::vector<float>& x,
                              const std::vector<float>& gamma,
                              const std::vector<float>& beta,
                              int rows, int width, bool softmax, float eps) {
  std::vector<double> result(x.size());
  for (int r = 0; r < rows; ++r) {
    const size_t base = static_cast<size_t>(r) * width;
    if (softmax) {
      double m = x[base];
      for (int c = 1; c < width; ++c) m = std::max(m, double(x[base + c]));
      double sum = 0.0;
      for (int c = 0; c < width; ++c) sum += std::exp(double(x[base + c]) - m);
      for (int c = 0; c < width; ++c) result[base + c] = std::exp(double(x[base + c]) - m) / sum;
    } else {
      double mean = 0.0, variance = 0.0;
      for (int c = 0; c < width; ++c) mean += x[base + c];
      mean /= width;
      for (int c = 0; c < width; ++c) {
        const double diff = x[base + c] - mean;
        variance += diff * diff;
      }
      variance /= width;
      for (int c = 0; c < width; ++c) {
        result[base + c] = (x[base + c] - mean) / std::sqrt(variance + eps) * gamma[c] + beta[c];
      }
    }
  }
  return result;
}

void run_case(int rows, int width, bool softmax, bool benchmark) {
  constexpr float eps = 1e-5f;
  const size_t count = static_cast<size_t>(rows) * width;
  const size_t bytes = count * sizeof(float);
  std::vector<float> x(count), actual(count), gamma(width), beta(width);
  for (int c = 0; c < width; ++c) {
    gamma[c] = 0.5f + (c % 5) * 0.25f;
    beta[c] = (c % 3 - 1) * 0.1f;
    for (int r = 0; r < rows; ++r) {
      x[static_cast<size_t>(r) * width + c] =
          r % 3 == 0 ? std::sin(float(c)) * 3.0f :
          r % 3 == 1 ? 7.0f : 1000.0f + float(c % 17 - 8);
    }
  }
  float *dx = nullptr, *dy = nullptr, *dg = nullptr, *db = nullptr;
  CUDA_CHECK(cudaMalloc(&dx, bytes));
  CUDA_CHECK(cudaMalloc(&dy, bytes));
  CUDA_CHECK(cudaMalloc(&dg, width * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, width * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dg, gamma.data(), width * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, beta.data(), width * sizeof(float), cudaMemcpyHostToDevice));

  auto launch = [&]() {
    if (softmax) row_softmax<<<rows, THREADS>>>(dx, dy, width);
    else row_layernorm<<<rows, THREADS>>>(dx, dg, db, dy, width, eps);
    CUDA_CHECK(cudaGetLastError());
  };
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(actual.data(), dy, bytes, cudaMemcpyDeviceToHost));
  const auto expected = reference(x, gamma, beta, rows, width, softmax, eps);
  double max_error = 0.0;
  const double atol = softmax ? 2e-6 : 2e-4;
  constexpr double rtol = 1e-3;
  for (size_t i = 0; i < count; ++i) {
    const double error = std::abs(double(actual[i]) - expected[i]);
    max_error = std::max(max_error, error);
    if (!std::isfinite(actual[i]) || error > atol + rtol * std::abs(expected[i])) {
      std::cerr << "FAIL at row=" << i / width << " col=" << i % width
                << " got=" << actual[i] << " expected=" << expected[i] << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
  if (softmax) {
    for (int r = 0; r < rows; ++r) {
      double sum = 0.0;
      for (int c = 0; c < width; ++c) sum += actual[static_cast<size_t>(r) * width + c];
      if (std::abs(sum - 1.0) > 1e-4) {
        std::cerr << "FAIL: probability row sum=" << sum << '\n';
        std::exit(EXIT_FAILURE);
      }
    }
  }
  std::cout << (softmax ? "softmax" : "layernorm") << " rows=" << rows
            << " width=" << width << " PASS max_abs_error=" << max_error;

  if (benchmark) {
    for (int i = 0; i < 20; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << " device_batch_mean_us=" << ms * 1000.0f / 100;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  std::cout << '\n';
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  CUDA_CHECK(cudaFree(dg));
  CUDA_CHECK(cudaFree(db));
}

int main(int argc, char** argv) {
  const bool benchmark = argc == 2 && std::string(argv[1]) == "--benchmark";
  const bool profile = argc == 2 && std::string(argv[1]) == "--profile";
  if (argc > 1 && !benchmark && !profile) {
    std::cerr << "usage: row_ops [--benchmark|--profile]\n";
    return EXIT_FAILURE;
  }
  if (profile) {
    run_case(128, 1024, true, false);
    run_case(128, 1024, false, false);
    return EXIT_SUCCESS;
  }
  for (int width : {1, 3, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 4097}) {
    run_case(3, width, true, false);
    run_case(3, width, false, false);
  }
  if (benchmark) {
    int device = 0;
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    std::cout << "device=" << props.name << " (device timing excludes allocations/copies)\n";
    run_case(128, 1024, true, true);
    run_case(128, 1024, false, true);
  }
  return EXIT_SUCCESS;
}
