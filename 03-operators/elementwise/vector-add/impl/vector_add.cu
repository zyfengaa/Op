// Vector Add：把 CPU 循环搬到 GPU 的最小完整例子。
//
// 阅读顺序：先读同级的 walkthrough.md（有手算和反例），再回到本文件对照。
// 本文件只解释每一行的机制；推导、性能估算和错误案例在 walkthrough 里。
//
// 本文件的代码按"单个线程负责一个元素"写，刻意不做向量化、不做 grid-stride，
// 目的是让第一次读 CUDA 的人能一行一行对上。生产实现见 assets/cuda/row_ops.cu。

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

// 每个 CUDA API 都返回 cudaError_t。不检查就直接往下走，失败会推迟到更难定位的地方爆发。
// 这个包装让失败时打印出"是哪一次调用失败"，而不是只给一个错误码。
void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    std::cerr << expression << ": " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

// 用宏包一层，是为了让 #expr 把调用表达式本身变成字符串传进去。
// 于是报错里会带上 "cudaMalloc(&d_a, bytes)" 这样的原文，直接指出出错的那一行。
#define CUDA_CHECK(expr) check_cuda((expr), #expr)

// __global__ 的含义：这个函数由 host（CPU）发起调用，在 device（GPU）上执行。
// 它是 kernel，只能用 <<<grid, block>>> 语法发射，不能像普通函数那样直接调用。
//
// 三个指针参数都是设备地址（device pointer），不是 std::vector 的宿主地址。
// 把它们搞混是新手最常见的崩溃：宿主指针传给 kernel 会触发非法访问。
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  // 这一行是整个 CUDA 编程模型的核心，值得背下来。
  //
  //   blockIdx.x  —— 当前 block 在 grid 里的编号（第几个 block）
  //   blockDim.x  —— 每个 block 有多少线程（发射时指定，这里是 256）
  //   threadIdx.x —— 当前线程在它所属 block 内的编号（0 .. blockDim.x-1）
  //
  // 三者组合，把二维的"第几个块 × 块内第几个"展平成一维的逻辑元素下标 i。
  // 结果是：每个有效线程恰好负责 C[i] 这一个元素，没有两个线程写同一个输出，
  // 所以这个 kernel 不需要任何同步。
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  // 边界保护。它不是"防御性编程"，是必需的：grid 用 ceil 发射（见下方 grid_size），
  // 除不尽时最后一个 block 会多出若干线程，它们的 i 落在 [n, grid*block) 区间内。
  //
  // 删掉这个 if 的后果：那些线程会读 A[i]/B[i] 并写 C[i]，全部越界。
  // 症状是"尾部几个输出保持原值或变成垃圾"——这正是故障题 VA01 要修的东西。
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

}  // namespace

int main() {
  // 刻意选一个不能被 256 整除的长度，这样每个 block 的尾部处理都会被真正执行到。
  // 如果取 n = 100096（256 的整数倍），上面那个 if 永远不会为假，VA01 这类 bug 就测不出来。
  constexpr int n = 100003;  // intentionally not divisible by block size

  // 字节数 vs 元素数：n 是元素个数，cudaMalloc 要的是字节数，必须乘 sizeof(float)。
  // 先 static_cast<size_t> 再相乘，避免在大张量上把乘法截断在 32 位里。
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  // 宿主侧数据。c 初始化为 0，这样万一某个位置没被 kernel 写到，会留下可见的 0 而不是随机值。
  std::vector<float> a(n), b(n), c(n, 0.0f);

  // 用随 i 变化的周期输入，而不是全 1 之类的常数。
  // 常数输入有个隐蔽的坑：读错相邻位置也常常得到同一个结果，索引置换类的 bug 就藏住了。
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i % 97 - 48) * 0.25f;
    b[i] = static_cast<float>(i % 31 - 15) * 0.5f;
  }

  // 设备指针。初始化为 nullptr，这样误用能立刻看出来是空指针而不是野指针。
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  // cudaMalloc 只在设备上划出地址，不搬运任何数据。
  // 这一步之后 d_a/d_b 里的内容是未定义的，必须靠下面的 H2D 拷贝填入。
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  // host → device 拷贝。这是显式内存模型的代价：CPU 的 std::vector 和 GPU 的显存是两个地址空间，
  // 数据不会自动过去。教学代码用同步版 cudaMemcpy，拷贝期间 host 阻塞。
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

  constexpr int block_size = 256;

  // ceil 除法：(n + block - 1) / block。
  // 整数除法会向下取整，所以要先把被除数补到 block 的整数倍。
  // 写成 n / block_size 会漏掉最后一截 —— 那正是 VA01 的症状来源。
  const int grid_size = (n + block_size - 1) / block_size;

  // <<<grid, block>>> 是 kernel 发射语法，不是函数调用。
  // 它配置好 grid 和 block 的维度后立即返回 —— launch 是异步的。
  vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

  // 因为 launch 异步，错误不会在这里抛出，所以要主动查一次。
  // 这个顺序很关键：cudaGetLastError 必须紧跟 launch，中间插别的调用会把错误状态清掉。
  // 典型能抓到的错误是 "invalid configuration argument"（block 超过 1024 线程之类）。
  CUDA_CHECK(cudaGetLastError());

  // 阻塞直到设备把之前排队的活干完。
  // 教学程序里它有两个作用：一是让后面的 D2H 拷贝能读到完整结果，
  // 二是让 kernel 执行期的错误（越界、非法地址）在这里暴露出来。
  // 生产流水线通常不用全设备同步，而是靠同一 stream 内的先后顺序 + 更窄的同步点。
  CUDA_CHECK(cudaDeviceSynchronize());

  // device → host，把结果取回宿主内存才能断言。
  CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  // 对拍：逐元素和 CPU 公式 a[i] + b[i] 比较。
  // 报错时打印出错的下标和两边的值，而不是只说一句 fail —— 下标本身就能提示是哪一类 bug
  // （尾部出错 → ceil 或 guard 问题；开头出错 → 索引映射问题）。
  for (int i = 0; i < n; ++i) {
    const float expected = a[i] + b[i];
    // isfinite 先挡住 NaN：NaN 参与的 abs(x - expected) > tol 会得到 false，
    // 只看容差比较会把 NaN 误判为"通过"。
    if (!std::isfinite(c[i]) || std::abs(c[i] - expected) > 1e-6f) {
      std::cerr << "vector add correctness: FAIL at " << i
                << " got=" << c[i] << " expected=" << expected << '\n';
      return EXIT_FAILURE;
    }
  }
  std::cout << "vector add correctness: PASS\n";

  // 设备内存不受 C++ 析构管理，必须手动释放。漏掉在小程序里看不出问题，在循环调用的场景会耗尽显存。
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return EXIT_SUCCESS;
}
