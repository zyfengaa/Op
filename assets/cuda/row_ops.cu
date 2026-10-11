// Row-wise Softmax 与 LayerNorm：一个 block 负责一整行。
//
// 与 ../../03-operators/reduction/impl/block_reduce_sum.cu 的区别：
// 那一版是"一个 block 归约整个输入"，这一版是"一行一个 block，行内归约"——
// 更接近真实 kernel 的组织方式，因为 softmax/layernorm 都是按行独立的。
//
// Teaching implementation: contiguous FP32 rows; finite inputs; forward only.
// Run without arguments for correctness, or --benchmark for a device-time sample.
//
// 明确没做的事（读的时候不要以为有）：不支持非连续输入、不处理 NaN/Inf、
// 只有前向没有反向、没有向量化访存、没有针对 bank conflict 做 padding。

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// 一个 block 的线程数，同时也是 block_reduce 里 shared 数组的长度。
// 选用 2 的幂不是习惯问题，是上面树形归约的硬要求：stride 每轮除以 2，
// 非 2 的幂会让最后一轮漏掉元素，而且不报错、只是结果偏小。
constexpr int THREADS = 256;  // Power of two, required by the reduction below.

void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    std::cerr << expression << ": " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr)

// 块内归约，用模板参数在编译期决定是求和还是求最大值。
// 不用模板的话要么写两份几乎相同的代码（改一处忘另一处），
// 要么在循环里传一个运行时的比较函数（多一层间接，热点代码不该这么写）。
//
// 两种归约的差别不只在运算本身，还在**单位元**：
//   求和用 0.0f 作初值，最大值必须用 -inf 作初值。
//   最大值若误用 0.0f 作初值，全负输入的结果会变成 0 —— 故障题 RD02 就是这个。
//   注意这里 -inf 的初值不在本函数里，而是在调用方（row_softmax 用了 -CUDART_INF_F）。
//
// Every thread must call this helper. It ends with a barrier AFTER reading s[0]:
// without it, a fast thread could overwrite s[0] during the next reduction.
// （也就是说：本函数返回前有一个同步，调用方可以连续调用两次而不出错。）
template <bool MAXIMUM>
__device__ float block_reduce(float value, float* s) {
  const int t = threadIdx.x;
  s[t] = value;
  __syncthreads();

  // 树形归约，与 block_reduce_sum.cu 里同一套写法。
  // fmaxf 而不是 std::max：前者是设备端的浮点内建函数，后者在 device 代码里不一定可用。
  for (int step = blockDim.x / 2; step > 0; step /= 2) {
    if (t < step) {
      s[t] = MAXIMUM ? fmaxf(s[t], s[t + step]) : s[t] + s[t + step];
    }
    __syncthreads();
  }

  // 所有线程都读 s[0] 拿到同一个结果 —— 不只是让 0 号线程知道。
  // 因为下一步（比如 softmax 的第二阶段）每个线程都要用这个全局 max。
  const float result = s[0];

  // 这个同步的作用和上面那个不同：它是为了"让下一次调用本函数安全"。
  // 若没有它，跑得快的线程可能已经开始写下一轮的 s[t]，而慢的线程还在读 s[0]。
  // 同一个 shared 数组被反复复用时，这类"读后写"竞争是最隐蔽的一类 bug。
  __syncthreads();
  return result;
}

// 行 softmax：一个 block 处理一行，行内所有列由本块线程协作完成。
//
// 分三阶段，每一阶段都要完整扫一遍这一行：
//   1. 求行最大值 m
//   2. 求 sum(exp(x - m))
//   3. 逐元素写 exp(x - m) / sum
// 三遍扫描是有代价的（这一行的数据被读了三次），换来的好处是不必把整行留在寄存器或 shared 里 ——
// 行宽可以远大于一个 block 能同时持有的容量。
__global__ void row_softmax(const float* x, float* y, int width) {
  __shared__ float scratch[THREADS];
  const int t = threadIdx.x;

  // 一个 block 对应一行，所以行的起点是 blockIdx.x * width。
  // 用 size_t 而不是 int：width 可能很大，行偏移与元素数的乘积会溢出 32 位。
  const size_t base = static_cast<size_t>(blockIdx.x) * width;

  // Stage 1: each thread scans columns t, t+256, t+512, ...
  //
  // 初值是 -inf 而不是 0 —— 这是"最大值归约的单位元"。
  // 用 0 作初值的后果：当整行都是负数时，max 会停在 0，而 0 不属于这一行，
  // 后续 exp(x - 0) 在大负数上直接下溢成 0，整行除以 0 得到 NaN。
  // 这正是故障题 SM01 描述的症状（"普通全负输入通过，极负输入 NaN"）。
  float local_max = -CUDART_INF_F;
  for (int col = t; col < width; col += blockDim.x) {
    local_max = fmaxf(local_max, x[base + col]);
  }
  // 归约得到这一行的全局最大值 m（所有线程都会拿到同一个值）。
  const float m = block_reduce<true>(local_max, scratch);

  // Stage 2: aggregate shifted exponentials; inactive threads contribute zero.
  //
  // 减去 m 是数值稳定性的关键：exp 的自变量保持在 0 附近，
  // 否则 exp(1000) 直接溢出成 inf。数学上 exp(x-m)/sum(exp(x-m)) 与 exp(x)/sum(exp(x)) 等价，
  // 但后者在大输入下算不出来。这是"等价变换换数值稳定性"的典型手法。
  //
  // 没有分到列的线程（t >= width）保持 local_sum = 0，即加法单位元，不影响总和。
  float local_sum = 0.0f;
  for (int col = t; col < width; col += blockDim.x) {
    local_sum += expf(x[base + col] - m);
  }
  const float denominator = block_reduce<false>(local_sum, scratch);

  // Stage 3: re-read/recompute for simplicity instead of retaining a whole row.
  //
  // 这里把 x 重新读一遍、exp 重新算一遍，而不是在第一阶段就把整行缓存下来。
  // 这是有意用"多算一遍"换"少占资源"：缓存整行需要 width 个寄存器的量级，
  // 行宽一大就会溢出到 local memory（表现为速度骤降）。
  // 生产实现（如 FlashAttention）用另一种办法省掉这一遍：在线维护 m 和 l 并随 max 更新而重缩放累加器。
  for (int col = t; col < width; col += blockDim.x) {
    y[base + col] = expf(x[base + col] - m) / denominator;
  }
}

// 行 layernorm：与 softmax 同构（也是一个 block 一行），但需要两次统计量。
//
// 这里是朴素的"先算均值、再算方差"两遍扫描，而不是 Welford 在线算法。
// 两遍扫描的好处是写法直白、容易验证；代价是这一行的数据被读了两遍。
// 更关键的是：这个写法在数值上不如 Welford 稳（大均值、小方差时会出现灾难性抵消），
// 但配合下面的 double 参考实现和容差，教学场景够用。
// CPU 侧的 dataflow_lab.py --topic welford 演示了 Welford 的多块合并公式。
__global__ void row_layernorm(const float* x, const float* gamma,
                              const float* beta, float* y, int width, float eps) {
  __shared__ float scratch[THREADS];
  const int t = threadIdx.x;
  const size_t base = static_cast<size_t>(blockIdx.x) * width;

  // 第一遍：均值。初值 0 是加法单位元，正确。
  float local_sum = 0.0f;
  for (int col = t; col < width; col += blockDim.x) {
    local_sum += x[base + col];
  }
  // 除以 width 得到均值。注意归约结果是所有线程共享的，除法在每个线程上各做一次也无妨。
  const float mean = block_reduce<false>(local_sum, scratch) / width;

  // 第二遍：方差。必须先有 mean 才能算，所以这两遍无法合并 ——
  // 这是"两遍扫描"相对于 Welford 的固有代价。
  // 注意 gamma/beta 是长度为 width 的**按列**参数，不参与归约。
  float local_m2 = 0.0f;
  for (int col = t; col < width; col += blockDim.x) {
    const float diff = x[base + col] - mean;
    local_m2 += diff * diff;
  }
  const float variance = block_reduce<false>(local_m2, scratch) / width;

  // rsqrtf(x) 即 1/sqrt(x)，是设备端内建函数，比 1.0f/sqrtf(x) 快且精度足够。
  // 加 eps 有两个作用：避免方差为 0 时除以 0（常量行就是这种情况），
  // 以及抑制极小方差被浮点误差放大。eps 的取值要和 dtype 匹配，fp32 常用 1e-5。
  const float inv_std = rsqrtf(variance + eps);

  // 归一化 + 仿射变换，一步写出结果。gamma/beta 按列索引（col 而不是 t），
  // 因为它们是 per-channel 参数，与"哪个线程负责这一列"无关。
  for (int col = t; col < width; col += blockDim.x) {
    y[base + col] = (x[base + col] - mean) * inv_std * gamma[col] + beta[col];
  }
}

// CPU 参考实现，用 double 累加。
//
// 用 double 是刻意的：它比 GPU 上的 float 累加更接近数学真值，
// 这样"GPU 结果和参考结果的差"才主要反映 GPU 侧的误差，而不是参考实现自己的误差。
// 两边都用 float 累加时，误差会互相掩盖，也说不清是谁不准。
std::vector<double> reference(const std::vector<float>& x,
                              const std::vector<float>& gamma,
                              const std::vector<float>& beta,
                              int rows, int width, bool softmax, float eps) {
  std::vector<double> result(x.size());
  for (int r = 0; r < rows; ++r) {
    const size_t base = static_cast<size_t>(r) * width;
    if (softmax) {
      // 与 GPU 版同构：先求 max，再减 max 求 exp 的和。
      double m = x[base];
      for (int c = 1; c < width; ++c) m = std::max(m, double(x[base + c]));
      double sum = 0.0;
      for (int c = 0; c < width; ++c) sum += std::exp(double(x[base + c]) - m);
      for (int c = 0; c < width; ++c) result[base + c] = std::exp(double(x[base + c]) - m) / sum;
    } else {
      // 与 GPU 版同构：两遍扫描，先均值后方差。
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

  // 输入按行号模 3 分成三种，覆盖三类数值情形：
  //   r%3==0 —— 有正有负、量级正常（sin 产生 [-3, 3]）
  //   r%3==1 —— 整行同一个正常值 7.0（常量行：方差为 0，考验 eps 是否起作用）
  //   r%3==2 —— 整行都在 1000 附近（大偏移小差异：考验减 max / 减 mean 的数值稳定性，
  //            也是 SM01 "极负输入 NaN" 的镜像情形）
  // 只测正常量级的输入，softmax 和 layernorm 的稳定性问题都不会暴露。
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
  // gamma/beta 只有 width 个元素，不是 count 个 —— 它们按列共享，所有行复用同一份。
  CUDA_CHECK(cudaMalloc(&dg, width * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, width * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dg, gamma.data(), width * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, beta.data(), width * sizeof(float), cudaMemcpyHostToDevice));

  // 把发射包成 lambda，让 benchmark 阶段能重复调用同一组发射而不必复制代码。
  auto launch = [&]() {
    // grid 的第一个维度是 rows —— 一个 block 负责一行。这个"行=block"的映射
    // 是这两个 kernel 的适用前提：行宽不能超过一个 block 能协作处理的规模。
    if (softmax) row_softmax<<<rows, THREADS>>>(dx, dy, width);
    else row_layernorm<<<rows, THREADS>>>(dx, dg, db, dy, width, eps);
    CUDA_CHECK(cudaGetLastError());
  };
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(actual.data(), dy, bytes, cudaMemcpyDeviceToHost));
  const auto expected = reference(x, gamma, beta, rows, width, softmax, eps);

  // 误差判据用 atol + rtol * |expected| 的组合形式：
  //   纯 atol 在数值很大时过严（大数的浮点绝对误差天然更大）
  //   纯 rtol 在数值接近 0 时过松（相对误差在小数上没有意义）
  // 组合形式是数值比较的通行做法。两个系数按算子不同而不同：
  // softmax 输出恒在 [0,1] 且逐行归一，误差更小，所以 atol 取 2e-6；
  // layernorm 输出没有归一化约束，取 2e-4。
  double max_error = 0.0;
  const double atol = softmax ? 2e-6 : 2e-4;
  constexpr double rtol = 1e-3;
  for (size_t i = 0; i < count; ++i) {
    const double error = std::abs(double(actual[i]) - expected[i]);
    max_error = std::max(max_error, error);
    // isfinite 先挡住 NaN —— NaN 与任何数比较都是 false，
    // 只用上面的容差判断会把 NaN 误判为通过。
    if (!std::isfinite(actual[i]) || error > atol + rtol * std::abs(expected[i])) {
      // 报错时给出 row/col 而不只是线性下标：行号和列号能直接提示是哪一类问题
      // （整行错 → 归约或统计量问题；个别列错 → gamma/beta 索引问题）。
      std::cerr << "FAIL at row=" << i / width << " col=" << i % width
                << " got=" << actual[i] << " expected=" << expected[i] << '\n';
      std::exit(EXIT_FAILURE);
    }
  }

  // softmax 额外检查一个不变量：每一行的输出必须求和为 1。
  // 这是逐元素比较**测不出来**的一类错误 —— 比如整行同时偏大或同时偏小，
  // 逐元素看可能都在容差内，但概率分布的性质已经破坏了。
  // 每加一个这样的不变量检查，就多堵住一类漏检。
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
  // 把 max_error 打出来而不只打 PASS：同样通过的两个实现，误差量级可能差一个数量级，
  // 而这个差别在下一步优化时可能变成"有没有超出容差"的分水岭。
  std::cout << (softmax ? "softmax" : "layernorm") << " rows=" << rows
            << " width=" << width << " PASS max_abs_error=" << max_error;

  if (benchmark) {
    // 热身 20 次再计时，让 kernel 加载、指令缓存、频率调节等一次性效应先完成。
    for (int i = 0; i < 20; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    // 用 CUDA Event 测，而不是 wall clock：
    //   host 计时器围住 launch 时，测到的主要是发射排队的时间，不是 kernel 执行时间。
    // 采样 100 次取平均。这里报的是批平均而不是中位数或最小值，所以只适合做粗对比；
    // 要做严格比较应当多批次采样并报中位数与区间（见 04-performance/measurement）。
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
  // --profile 只跑一个固定规模（128×1024），供 ncu/nsys 采集单一配置下的 kernel。
  // 采集性能计数器时不要跑一长串规模：每换一个规模就是另一组参数，指标之间不可比。
  if (profile) {
    run_case(128, 1024, true, false);
    run_case(128, 1024, false, false);
    return EXIT_SUCCESS;
  }

  // 这 12 个宽度不是随便挑的，每一类都对应一种会出错的边界：
  //   1        —— 行宽小于 block，绝大多数线程无活可干
  //   3        —— 极小且非 2 的幂
  //   31 / 32 / 33 —— 围绕一个 warp 的宽度（32）：31 有线程空转，33 需要两个 warp
  //   255/256/257 —— 围绕 block 大小 THREADS
  //   1023/1024/1025 —— 围绕 4 个 warp，偏一行一列都会改变归约轮次
  //   4097     —— 大且为质数，保证各种 ceil 和 stride 的余数都非零
  // 另外 32 的倍数还与 shared memory 的 bank 有关：scratch[t] 在 t 跨 32 时可能落在同一个 bank
  // （bank conflict）。这个教学版本没有为此做 padding，因为 CPU 侧无法验证收益。
  // 只测 1024 这种"整齐"的宽度，上面这些路径全都走不到。
  for (int width : {1, 3, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 4097}) {
    run_case(3, width, true, false);
    run_case(3, width, false, false);
  }

  if (benchmark) {
    // 打印设备名：同一个时间数字在 4090 和 A100 上含义完全不同，
    // 性能数字离开了设备标识就不可解释。
    int device = 0;
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    std::cout << "device=" << props.name << " (device timing excludes allocations/copies)\n";
    // 明确标注计时范围：这里的 device_batch_mean_us **不包含** cudaMalloc 和 cudaMemcpy。
    // 这与端到端测出来的是两个不同的量，不能拿来互相比较，也不能拼成一个"总时间"。
    run_case(128, 1024, true, true);
    run_case(128, 1024, false, true);
  }
  return EXIT_SUCCESS;
}
