// Naive GEMM：每个线程算一个输出元素，没有任何数据复用优化。
//
// 这是优化阶梯的起点，不是终点。它存在的意义是提供一个"正确但慢"的基线，
// 后面 v2-tiled 的每一步收益都拿它当参照。
//
// 阅读顺序：先读 ../v2-tiled/tiled_gemm.cu，再回来对比这一版差在哪。
// 两版放在一起读，才看得出 tiling 到底解决了什么。

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

// 宏包装的意义：让 #expr 把调用表达式本身变成字符串，报错时能指出是哪一行失败的。
#define CUDA_CHECK(expr) check_cuda((expr), #expr)

// 每个线程负责 C 的一个元素。
//
// 与 vector_add 的区别：这里用了二维的 block 和 grid，因为输出本身是二维的。
// 用二维索引（而不是把 C 展平成一维）的好处是寻址和数学写法能对上，
// 读代码时不用在脑子里做除法。
__global__ void naive_gemm_kernel(const float* a, const float* b, float* c,
                                  int m, int n, int k) {
  // 二维展平：y 方向对应输出行，x 方向对应输出列。
  // 注意 threadIdx.y 管行、threadIdx.x 管列 —— 这与 x 是"最快变化的维度"这个惯例一致，
  // 也让相邻的 threadIdx.x 映射到相邻的内存列，利于合并访存。
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  // 越界直接 return，而不是用 if 把计算包起来。
  // 效果等价（这个线程不写任何输出），但省掉一层缩进，在更复杂的 kernel 里可读性更好。
  // 代价是同一个 warp 里如果有线程提前 return，它们只是被 mask 掉，不会让别的线程变快。
  if (row >= m || col >= n) {
    return;
  }

  // 内层对 k 的循环：每次迭代读 a 的一个元素和 b 的一个元素，做一次乘加。
  //
  // 这正是这一版慢的原因，也是 tiling 要解决的问题：
  //   a[row * k + inner] 在这一行内被 n 个不同线程各读一次（col 不同，row、inner 相同）
  //   b[inner * n + col] 在这一列内被 m 个不同线程各读一次
  // 也就是说每个输入元素都被从 global memory 重复读取了几十次。
  // 没有 shared memory 缓存，这些重复读全部落到 L1/L2/DRAM 上。
  float acc = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    acc += a[row * k + inner] * b[inner * n + col];
  }

  // row-major 索引：C 是 m×n，第 row 行第 col 列的线性下标是 row * n + col。
  // 注意乘的是 n（C 的宽度），不是 k。写错这一处是最常见的 GEMM bug。
  c[row * n + col] = acc;
}

// CPU 参考实现。三层朴素循环，职责是提供金标准，刻意不追求速度。
// 与 v1/v2 两个 GPU 版本逐元素对拍，容差 1e-4（见 main）。
std::vector<float> cpu_gemm(const std::vector<float>& a,
                            const std::vector<float>& b, int m, int n, int k) {
  std::vector<float> c(static_cast<size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      for (int inner = 0; inner < k; ++inner) {
        // 同一个表达式 a[row*k+inner] * b[inner*n+col]，和 GPU 版本逐字对应。
        // 两边写法一致，对拍失败时才能确定是执行差异而不是公式差异。
        c[row * n + col] += a[row * k + inner] * b[inner * n + col];
      }
    }
  }
  return c;
}

}  // namespace

int main() {
  // 三个维度都刻意取质数，相对 16 都不能整除。
  // 这样 grid 的 ceil 除法、kernel 里的越界 return 都会被真正执行到。
  // 全取 16 的整数倍会让这些边界路径永远不被覆盖，bug 就藏起来了。
  constexpr int m = 37;
  constexpr int n = 29;
  constexpr int k = 41;

  // 三个矩阵的字节数各不相同，因为形状不同：A 是 m×k，B 是 k×n，C 是 m×n。
  // 混淆这三个形状是 GEMM 实现里最常见的错误来源。
  const size_t a_bytes = static_cast<size_t>(m) * k * sizeof(float);
  const size_t b_bytes = static_cast<size_t>(k) * n * sizeof(float);
  const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
  std::vector<float> a(m * k), b(k * n), c(m * n);

  // 周期性的小整数输入，避免全常数导致"读错位置也得到同样结果"而掩盖索引 bug。
  for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
  for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<float>(static_cast<int>(i % 7) - 3);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
  CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
  CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), a_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), b_bytes, cudaMemcpyHostToDevice));

  // 16×16 = 256 个线程一个 block。
  //
  // 为什么是 16×16 而不是 256×1：一个 block 覆盖输出的一小块方阵，
  // 相邻的 x 方向线程映射到相邻的列，同一 warp 的访存落在连续地址上（合并访问）。
  // 换成 256×1 也能算对，但访存模式会差很多。
  dim3 block(16, 16);

  // 二维的 ceil 除法，两个方向各自向上取整。
  // 注意 grid.x 对应的是 n（列数），grid.y 对应的是 m（行数）—— 顺序和 block 一致，
  // 和上面 row/col 的求法对应。这两处顺序写反是经典 bug。
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  naive_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

  // 拿 CPU 参考实现的结果当期望值，而不是反过来。
  // 如果先有 GPU 结果再让参考实现去凑，两边一起错的时候测试照样全绿。
  const auto expected = cpu_gemm(a, b, m, n, k);

  // 容差 1e-4 比 vector_add 的 1e-6 宽，因为 GEMM 累加 k=41 次，
  // 浮点误差随累加次数增长。容差要跟 dtype 和累加深度匹配，不是越小越好。
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
