// Executable optimization ladders for Vector Add and Sum. No injected defects here.
// Every version uses the same inputs/reference. --benchmark times the largest case.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

constexpr int T = 256;
void checked(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    std::cerr << expression << ": " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}
#define CHECK(expr) checked((expr), #expr)

__global__ void add_v0(const float* a, const float* b, float* out, int n) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
__global__ void add_v1(const float* a, const float* b, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
__global__ void add_v2(const float* a, const float* b, float* out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    out[i] = a[i] + b[i];
}

__global__ void sum_v0(const float* x, float* out, int n) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float acc = 0;
    for (int i = 0; i < n; ++i) acc += x[i];
    out[0] = acc;
  }
}
__global__ void sum_blocks(const float* x, float* partial, int n) {
  __shared__ float s[T];
  const int t = threadIdx.x;
  float acc = 0;
  for (int i = blockIdx.x * blockDim.x + t; i < n; i += gridDim.x * blockDim.x)
    acc += x[i];
  s[t] = acc;
  __syncthreads();
  for (int stride = T / 2; stride; stride /= 2) {
    if (t < stride) s[t] += s[t + stride];
    __syncthreads();
  }
  if (t == 0) partial[blockIdx.x] = s[0];
}

void run_case(int n, bool benchmark) {
  if (n == 0) {
    std::cout << "N=0: host returns empty add and sum=0; no zero-grid launch\n";
    return;
  }
  std::vector<float> a(n), b(n), actual(n);
  double expected_sum = 0;
  for (int i = 0; i < n; ++i) {
    a[i] = (i % 13 - 6) * 0.25f;
    b[i] = (i % 7 - 3) * 0.5f;
    expected_sum += a[i];
  }
  const int all_blocks = (n + T - 1) / T;
  const int capped_blocks = std::min(all_blocks, 64);
  float *da = nullptr, *db = nullptr, *dy = nullptr, *partial = nullptr;
  CHECK(cudaMalloc(&da, n * sizeof(float)));
  CHECK(cudaMalloc(&db, n * sizeof(float)));
  CHECK(cudaMalloc(&dy, n * sizeof(float)));
  CHECK(cudaMalloc(&partial, capped_blocks * sizeof(float)));
  CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  for (int op = 0; op < 2; ++op) {
    for (int version = 0; version < 3; ++version) {
      auto launch = [&]() {
        if (op == 0) {
          if (version == 0) add_v0<<<1, 1>>>(da, db, dy, n);
          if (version == 1) add_v1<<<all_blocks, T>>>(da, db, dy, n);
          if (version == 2) add_v2<<<capped_blocks, T>>>(da, db, dy, n);
        } else {
          if (version == 0) sum_v0<<<1, 1>>>(da, dy, n);
          if (version == 1) sum_blocks<<<1, T>>>(da, dy, n);
          if (version == 2) {
            sum_blocks<<<capped_blocks, T>>>(da, partial, n);
            CHECK(cudaGetLastError());
            // Same stream orders the producer and consumer kernels.
            sum_blocks<<<1, T>>>(partial, dy, capped_blocks);
          }
        }
        CHECK(cudaGetLastError());
      };
      launch();
      CHECK(cudaDeviceSynchronize());
      const int output_size = op == 0 ? n : 1;
      CHECK(cudaMemcpy(actual.data(), dy, output_size * sizeof(float), cudaMemcpyDeviceToHost));
      for (int i = 0; i < output_size; ++i) {
        const double expected = op == 0 ? double(a[i] + b[i]) : expected_sum;
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected) > 1e-5) {
          std::cerr << "FAIL op=" << op << " v=" << version << " n=" << n << " index=" << i << '\n';
          std::exit(EXIT_FAILURE);
        }
      }
      std::cout << (op == 0 ? "add" : "sum") << " v" << version << " N=" << n << " PASS";
      if (benchmark) {
        for (int i = 0; i < 5; ++i) launch();
        CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i) launch();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        std::cout << " batch_mean_us=" << ms * 1000 / 20;
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
      }
      std::cout << '\n';
    }
  }
  CHECK(cudaFree(da));
  CHECK(cudaFree(db));
  CHECK(cudaFree(dy));
  CHECK(cudaFree(partial));
}

int main(int argc, char** argv) {
  bool benchmark = argc == 2 && std::string(argv[1]) == "--benchmark";
  if (argc > 1 && !benchmark) {
    std::cerr << "usage: operator_ladders [--benchmark]\n";
    return EXIT_FAILURE;
  }
  for (int n : {0, 1, 255, 256, 257, 4097, 65539})
    run_case(n, benchmark && n == 65539);
  return EXIT_SUCCESS;
}
