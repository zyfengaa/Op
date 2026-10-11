// Tiled GEMM：用 shared memory 做数据复用，比 v1-naive 少读几十倍的全局内存。
//
// 这是优化阶梯的第二级。v1 里每个输入元素被从 global memory 重复读了 n 次或 m 次，
// 这一版把一小块（tile）先协作搬进 shared memory，块内线程再反复消费它。
//
// 阅读方法：和 ../v1-naive/naive_gemm.cu 对比着读，不要单独看这一版 ——
// 单独看只能看到"多了 shared memory 和两个同步"，看不出它们换来了什么。

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
// tile 边长。TILE×TILE = 256 个线程，正好对应一个 block 的线程数。
// 这个"一个线程搬一个元素"的对应关系是下面协作装载能写成一行赋值的前提。
constexpr int TILE = 16;

void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    std::cerr << expression << ": " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr)

// 一个 block 算 C 的一个 TILE×TILE 输出块。
//
// 核心思想：这个 block 的 256 个线程，对 A 的一行片段要读 TILE 次、对 B 的一列片段也要读 TILE 次。
// 与其每次都去 global memory 取，不如先协作搬进 shared memory 再消费。
// shared memory 的延迟比 global 低一到两个数量级，且同一 block 内共享。
//
// 复用的确切数字（以这一步为例）：
//   搬入 shared 的 A tile 有 TILE×TILE 个元素，被消费 TILE 次（每个列方向线程各用一遍）
//   所以 global memory 的读取量大约降到 1/TILE —— 代价是两次 __syncthreads() 和 shared 容量。
__global__ void tiled_gemm_kernel(const float* a, const float* b, float* c,
                                  int m, int n, int k) {
  // shared memory：同一个 block 内所有线程可见，block 结束后失效。
  // 不加 __shared__ 的话这两个数组会落在每个线程私有的寄存器/本地内存里，
  // 那就退化成"每个线程各存一份自己的副本"，完全失去共享的意义。
  __shared__ float tile_a[TILE][TILE];
  __shared__ float tile_b[TILE][TILE];

  // 这个线程负责的输出元素（全局坐标）。
  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  // 累加器放在寄存器里（每个线程私有），整个 k 维度上一直累加，
  // 只在最后写回一次 global memory。这是 GEMM 优化的通用做法。
  float acc = 0.0f;

  // k 维度要分几块。同样用 ceil，因为 k 通常不是 TILE 的整数倍。
  const int k_tiles = (k + TILE - 1) / TILE;

  for (int tile = 0; tile < k_tiles; ++tile) {
    // 本线程要搬的两个元素在当前 tile 里的全局坐标：
    //   A 的列 = tile 起点 + 本线程的 x 编号
    //   B 的行 = tile 起点 + 本线程的 y 编号
    // 注意 A 用 x、B 用 y —— 这个交叉是有意的，见下面两行装载的说明。
    const int a_col = tile * TILE + threadIdx.x;
    const int b_row = tile * TILE + threadIdx.y;

    // 协作装载：256 个线程，每个线程搬 1 个 A 元素和 1 个 B 元素。
    //
    // A 的装载：本线程（row, a_col）→ 存到 tile_a[threadIdx.y][threadIdx.x]
    //   即"我这一行的第 a_col 列"，落到 shared 里我这一行对应的位置。
    // B 的装载：（b_row, col）→ 存到 tile_b[threadIdx.y][threadIdx.x]
    //
    // 越界的部分填 0，而不是跳过不写。原因有两个：
    //   1. 累加循环里 0 不贡献任何值，等价于该位置不存在（这就是"加法单位元"）。
    //      与归约里选 identity 元素是同一个道理，选错的症状见故障题 RD02。
    //   2. 更重要：下面有 __syncthreads()，所有线程都必须执行到它。
    //      如果这里用 if 把装载包起来，越界线程就会跳过同步点，导致死锁或读到脏数据。
    //      "输出是否有效"和"是否需要参与协作装载"是两件事 —— 这是故障题 GM01 的考点。
    tile_a[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

    // 第一个同步点：等所有线程都搬完，才能开始消费。
    // 少了它，先跑到这里的线程可能读到别人还没写完的 shared 内容。
    __syncthreads();

    // 消费本 tile：每个线程沿着 inner 方向走一遍，取 A 的"我这一行"和 B 的"我这一列"。
    // 这两次读取都命中 shared memory，不再碰 global。
    #pragma unroll
    for (int inner = 0; inner < TILE; ++inner) {
      acc += tile_a[threadIdx.y][inner] * tile_b[inner][threadIdx.x];
    }

    // 第二个同步点：等所有线程都消费完，下一轮才能覆盖这两个 tile。
    // 少了它，跑得快的线程会在别人还在读的时候写坏 shared 内容 ——
    // 这类 bug 表现为结果偶尔不对、换机器或换 block size 就不复现，比崩溃难查得多。
    // Do not overwrite a tile while another thread still consumes it.
    __syncthreads();
  }

  // 只有落在输出范围内的线程才写回。
  // 注意这个判断在循环之外 —— 循环内部所有线程都必须全程参与，理由见上面的装载注释。
  if (row < m && col < n) {
    c[row * n + col] = acc;
  }
}

// CPU 参考实现，与 v1-naive 里的完全一致。刻意保持两版公式逐字相同，
// 这样对拍失败时能确定是执行差异而不是公式差异。
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
  // 三个维度都相对 TILE=16 有余数，且互不相等。
  // 这让 k 方向的 tile 循环、两个方向上的 tile 填充、以及最后的写回判断全部被走到。
  // All three dimensions have tails relative to TILE.
  constexpr int m = 37, n = 29, k = 41;

  // 三个矩阵形状不同，字节数也不同：A 是 m×k，B 是 k×n，C 是 m×n。
  const size_t a_bytes = static_cast<size_t>(m) * k * sizeof(float);
  const size_t b_bytes = static_cast<size_t>(k) * n * sizeof(float);
  const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
  std::vector<float> a(m * k), b(k * n), c(m * n);

  // 周期性小整数输入，避免常数输入掩盖索引置换类 bug。
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

  // block 是 2D 的 (TILE, TILE)，和 kernel 里的 threadIdx.x/y 对应：
  // x 走输出列方向，y 走输出行方向。
  dim3 block(TILE, TILE);

  // grid 的 x 覆盖列（n），y 覆盖行（m），各自 ceil 到 TILE 的整数倍。
  // 两个方向顺序写反是这个 kernel 最经典的一类错误。
  dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  tiled_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

  // 期望值来自 CPU 参考实现。注意即使 v2 理论上更快，正确性基准仍然是 v1/cpu 那一侧。
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
