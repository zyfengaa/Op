// 可执行的优化阶梯：Vector Add 和 Sum 各三档，从"完全不用并行"做到"两阶段归约"。
//
// 与故障题库的区别：这个文件里**没有注入任何缺陷**，每一档都是正确实现。
// 它的用途是对照着看"同一件事用不同的并行方式写，会差在哪里"。
//
// 阅读方法：v0 → v1 → v2 顺着读，看并行度是怎么一步步加上去的。
// 三档用完全相同的输入和参考值，所以数字之间可以直接比。

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

constexpr int T = 256;
// 与其它文件里的 CUDA_CHECK 是同一个东西，只是这里名字短一些。
void checked(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    std::cerr << expression << ": " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}
#define CHECK(expr) checked((expr), #expr)

// ── Add 的三个版本 ────────────────────────────────────────────────

// v0：只用**一个线程**做完整件事。
// 这不是"写得差"，而是刻意保留的对照组 —— 它代表把 CPU 循环原样搬过来的思路，
// 在 GPU 上完全没用到并行能力。整个 grid 里只有一个线程在干活，其余全部空转。
__global__ void add_v0(const float* a, const float* b, float* out, int n) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

// v1：一个线程负责一个元素。
// 这是最直接的并行化：把循环下标 i 换成"由线程编号算出来的 i"。
// grid 必须覆盖到 n，所以发射的 block 数是 ceil(n / T)。
// 边界判断必需，因为最后一个 block 会多出无效线程（详见 vector_add.cu 和故障题 VA01）。
__global__ void add_v1(const float* a, const float* b, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

// v2：grid-stride 循环。
// 每个线程处理多个元素，步长是"整个 grid 的线程总数"。
//
// 与 v1 的关键区别：发射多少个 block 不再由 n 决定，可以人为限制。
// 好处是 n 很大时不必发射海量 block（发射本身有开销），
// 代价是每个线程的工作量变成不确定的，负载均衡要考虑。
// 注意步长是 gridDim.x * blockDim.x（跨 block），不是 blockDim.x（仅块内）。
__global__ void add_v2(const float* a, const float* b, float* out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    out[i] = a[i] + b[i];
}

// ── Sum 的两个版本 ────────────────────────────────────────────────

// 单线程求和。同样是对照组，同时也是"归约结果应该是什么"的最直白定义。
__global__ void sum_v0(const float* x, float* out, int n) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float acc = 0;
    for (int i = 0; i < n; ++i) acc += x[i];
    out[0] = acc;
  }
}

// 归约：每个 block 先用 grid-stride 算出自己的局部和，再做块内树形归约，
// 最后由 0 号线程把本块的结果写到 partial[blockIdx.x]。
//
// 这个 kernel 被用了两次，扮演两个不同角色（见 run_case 里的发射）：
//   第一次：输入是原始数据，输出是每个 block 一个局部和
//   第二次：输入是上一步的局部和数组，用一个 block 把它们归约成一个标量
// 同一个 kernel 能复用，是因为它只要求"block 数 = 输出元素数"这一条，
// 而两次调用的规模恰好都满足。
__global__ void sum_blocks(const float* x, float* partial, int n) {
  // 每个 block 私有的临时数组，用于块内归约。
  __shared__ float s[T];
  const int t = threadIdx.x;

  // 局部累加：每个线程处理若干个元素。注意初值是 0，
  // 也就是加法单位元 —— 没有分配到这个线程的位置，贡献 0，不影响结果。
  // 选错单位元的后果见故障题 RD02（全负数时最大值变成零）。
  float acc = 0;
  for (int i = blockIdx.x * blockDim.x + t; i < n; i += gridDim.x * blockDim.x)
    acc += x[i];

  s[t] = acc;
  __syncthreads();

  // 树形归约：每轮把问题规模减半。
  // 每轮之后都要同步，因为下一轮读的是别的线程刚写的 s[t + stride]。
  // T 是 2 的幂，所以 stride 能一路整除到 1；换成非 2 的幂会漏掉元素。
  for (int stride = T / 2; stride; stride /= 2) {
    if (t < stride) s[t] += s[t + stride];
    __syncthreads();
  }

  // 只有 0 号线程写回。partial 的每个位置对应一个 block。
  if (t == 0) partial[blockIdx.x] = s[0];
}

void run_case(int n, bool benchmark) {
  // N=0 必须在 host 端提前返回，不能发射一个 0 block 的 grid。
  // 发射 0 个 block 是未定义行为（旧版 CUDA 会直接报错），
  // 而且"求和为空集"在数学上应当是 0，这个语义要由 host 显式给出。
  if (n == 0) {
    std::cout << "N=0: host returns empty add and sum=0; no zero-grid launch\n";
    return;
  }

  std::vector<float> a(n), b(n), actual(n);

  // 参考和在 host 上用 double 累加，比 GPU 上的 float 累加更准，
  // 所以它适合当基准；反过来（用 GPU 结果当基准）会让两边一起错。
  double expected_sum = 0;
  for (int i = 0; i < n; ++i) {
    a[i] = (i % 13 - 6) * 0.25f;
    b[i] = (i % 7 - 3) * 0.5f;
    expected_sum += a[i];
  }

  // 两个不同的 grid 规模：
  //   all_blocks  —— v1 用，必须覆盖每个元素
  //   capped_blocks —— v2 和归约用，封顶 64 个 block，靠 grid-stride 循环覆盖剩余部分
  // 两者的差别正是 v1 和 v2 的设计取舍。
  const int all_blocks = (n + T - 1) / T;
  const int capped_blocks = std::min(all_blocks, 64);

  float *da = nullptr, *db = nullptr, *dy = nullptr, *partial = nullptr;
  CHECK(cudaMalloc(&da, n * sizeof(float)));
  CHECK(cudaMalloc(&db, n * sizeof(float)));
  CHECK(cudaMalloc(&dy, n * sizeof(float)));

  // partial 只需要 capped_blocks 个 float —— 每个 block 产出一个局部和。
  CHECK(cudaMalloc(&partial, capped_blocks * sizeof(float)));
  CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  // 外层 op（0=add，1=sum）× 内层 version（0/1/2）的笛卡尔积，共 6 种组合。
  for (int op = 0; op < 2; ++op) {
    for (int version = 0; version < 3; ++version) {
      // 用 lambda 把"发射"包起来，是为了 benchmark 阶段能重复调用同一组发射，
      // 而不用把发射代码写两遍（写两遍就有两边不一致的风险）。
      auto launch = [&]() {
        if (op == 0) {
          if (version == 0) add_v0<<<1, 1>>>(da, db, dy, n);
          if (version == 1) add_v1<<<all_blocks, T>>>(da, db, dy, n);
          if (version == 2) add_v2<<<capped_blocks, T>>>(da, db, dy, n);
        } else {
          if (version == 0) sum_v0<<<1, 1>>>(da, dy, n);
          if (version == 1) sum_blocks<<<1, T>>>(da, dy, n);
          if (version == 2) {
            // 两阶段归约：先各块出局部和，再用一个 block 归约这些局部和。
            sum_blocks<<<capped_blocks, T>>>(da, partial, n);
            CHECK(cudaGetLastError());
            // 两次发射在同一个 stream 上，天然有先后顺序，
            // 所以第二阶段能安全读取第一阶段写出的 partial。
            // 换成不同 stream 就必须显式同步。
            // Same stream orders the producer and consumer kernels.
            sum_blocks<<<1, T>>>(partial, dy, capped_blocks);
          }
        }
        CHECK(cudaGetLastError());
      };
      launch();
      CHECK(cudaDeviceSynchronize());

      // add 的输出是 n 个元素，sum 的输出是 1 个标量 —— 取回的长度不同。
      const int output_size = op == 0 ? n : 1;
      CHECK(cudaMemcpy(actual.data(), dy, output_size * sizeof(float), cudaMemcpyDeviceToHost));

      // 期望值：add 逐元素和 CPU 公式比，sum 统一和 host 的 double 累加比。
      // 容差 1e-5 对 n 达到 65539 的 float 累加来说是合理量级 ——
      // 累加次数越多误差越大，容差不该设成与规模无关的固定小数。
      for (int i = 0; i < output_size; ++i) {
        const double expected = op == 0 ? double(a[i] + b[i]) : expected_sum;
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected) > 1e-5) {
          std::cerr << "FAIL op=" << op << " v=" << version << " n=" << n << " index=" << i << '\n';
          std::exit(EXIT_FAILURE);
        }
      }
      std::cout << (op == 0 ? "add" : "sum") << " v" << version << " N=" << n << " PASS";

      if (benchmark) {
        // 先热身 5 次再计时。第一次发射通常包含 kernel 加载、指令缓存填充等一次性开销，
        // 把它算进平均会让结果偏高，而且偏高多少取决于之前的执行历史。
        for (int i = 0; i < 5; ++i) launch();
        CHECK(cudaDeviceSynchronize());

        // 用 CUDA Event 而不是 host 计时器：
        // host 计时会把发射的排队时间也算进去，而且 launch 是异步的，
        // 不配合同步根本量不到 kernel 真正执行了多久。
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        // 重复 20 次取批平均，而不是报单次最小值 ——
        // 单次最小值反映的是最好情况，不是稳定表现。
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
  // 每个规模都测，覆盖四类边界：
  //   0      —— 空输入，走 host 提前返回那条路
  //   1      —— 单元素，grid-stride 循环只跑一轮
  //   255/256/257 —— block 大小 256 的两侧和正中间，覆盖尾部逻辑
  //   4097   —— 不能被 256 整除的中等规模
  //   65539  —— 大输入，且不是 256 的整数倍，用于 v1 和 v2 的差距显现
  // 只测 256 的整数倍会让所有尾部路径永远不被执行。
  for (int n : {0, 1, 255, 256, 257, 4097, 65539})
    run_case(n, benchmark && n == 65539);
  return EXIT_SUCCESS;
}
