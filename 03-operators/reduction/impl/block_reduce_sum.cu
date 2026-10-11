// Block 内归约：一个 block 把整个输入求和成一个标量。
//
// 这是第一次出现"多个线程协作算出一个结果"，也是本仓库里最陡的一段。
// 建议先跑无 GPU 的 ../../../assets/cpu/reduction_lab.py，看清两级归属和 identity 元素，
// 再回到本文件对照。
//
// 阅读难点：__syncthreads() 出现的位置和次数。少一个就错，且症状不稳定。

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
// （两步归约的完整例子见 ../../../assets/cuda/operator_ladders.cu 里的 sum_blocks v2。）
//
// 本 kernel 的输入规模刻意很小：一个 block、最多 256 个线程。
// 真实场景里输入远大于一个 block 能覆盖的范围，那时必须分两级做 ——
// 每块出自己的局部和，再用第二个 kernel（或原子操作）合并，否则要么装不下，要么并行度太低。
__global__ void block_sum_kernel(const float* input, float* output, int n) {
  // shared memory：同一个 block 内所有线程共享，块内用它交换中间结果。
  // 每个线程的局部和被写进 partial[tid]，随后所有线程合作把它归约成 partial[0]。
  __shared__ float partial[THREADS];
  const int tid = threadIdx.x;

  // 第一步：局部累加。每个线程跨步扫描，负责 tid, tid+256, tid+512, ... 这些位置。
  //
  // 这个写法一次性解决了 n 不是 THREADS 整数倍的情况：
  // 循环条件 i < n 让越界的线程自然少跑几轮，不需要额外的边界分支。
  // 完全分配不到元素的线程（tid >= n）保持 local = 0 ——
  // 也就是加法单位元，它们对总和贡献 0，不需要特殊处理。
  // 这正是故障题 RD02 考的点：单位元选错（例如 max 归约里选 0 而不是 -inf），
  // 全负输入时结果就会被那个 0 污染。
  float local = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    local += input[i];
  }
  partial[tid] = local;

  // 第一次同步：必须先让所有线程把 partial[tid] 写完，才能开始读别人写的位置。
  // 少了它，跑得快的线程会读到还没写入的 shared 内容（可能是上次残留的任意值）。
  __syncthreads();

  // 第二步：块内树形归约。每轮把参与归约的线程数减半，log2(256) = 8 轮做完。
  //
  // THREADS 必须是 2 的幂，否则 stride 一路除以 2 最后会落在 0 之前的某个非零值上，
  // 有一部分元素永远不被合并 —— 症状是"结果偏小"而不是报错，很难发现。
  // THREADS is a power of two. Every reduction step is block-wide.
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      partial[tid] += partial[tid + offset];
    }
    // 每轮之后都要同步：下一轮读的 tid + offset 是别的线程刚刚写过的位置。
    // 这个同步在 for 循环**内部**，且在 if 之外 ——
    // 所有线程都必须执行到它，不能把同步写进 if (tid < offset) 里面。
    __syncthreads();
  }

  // 归约结束后，partial[0] 就是全块的总和。
  // 只让 0 号线程写回，因为只有它是权威结果；其他线程的 partial 值已经没有意义。
  if (tid == 0) {
    output[0] = partial[0];
  }
}
}  // namespace

int main() {
  // 1003 相对 block 大小 256 有余数（1003 = 3×256 + 235），
  // 所以最后一个 block 的第 4 轮只有部分线程有活干 —— 尾部路径被真正走到。
  // 用 1024 这样的整数倍会让上面那个 `i < n` 的边界逻辑永远不生效。
  constexpr int n = 1003;  // tail relative to the block size
  std::vector<float> input(n);

  // 参考和在 host 上用 float 逐项累加，与 kernel 的累加顺序不同。
  // 这让容差必须留有余量（下面用 1e-3）—— 浮点加法不满足结合律，
  // 换个加法顺序就会得到略有差异的结果，这是"结果不一致"的常见来源，不是 bug。
  float reference = 0.0f;
  for (int i = 0; i < n; ++i) {
    input[i] = static_cast<float>((i % 11) - 5);
    reference += input[i];
  }

  float *d_input = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));

  // 输出只有一个 float：归约到标量，不是逐元素输出。
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  // <<<1, THREADS>>> —— 只发射 1 个 block、256 个线程。
  // 这是"教学版"的边界：再多一倍的输入，这一个 block 就得靠循环硬扛，效率会掉。
  block_sum_kernel<<<1, THREADS>>>(d_input, d_output, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float actual = 0.0f;
  CUDA_CHECK(cudaMemcpy(&actual, d_output, sizeof(float), cudaMemcpyDeviceToHost));

  // 容差 1e-3 而不是 1e-6：两边累加顺序不同，误差量级由累加深度决定。
  // 容差应当与 dtype 和归约规模匹配，取一个"与规模无关的固定小数"是常见的错误做法。
  if (!std::isfinite(actual) || std::abs(actual - reference) > 1e-3f) {
    std::cerr << "block reduction correctness: FAIL: " << actual
              << " vs " << reference << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "block reduction correctness: PASS\n";

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return EXIT_SUCCESS;
}
