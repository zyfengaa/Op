"""Print intermediate states that connect mathematics to kernel dataflow.

Run from the repo root: python assets/cpu/dataflow_lab.py
Finite Python floats only; timings are deliberately not presented as GPU evidence.
"""

import argparse
import math

from gemm_lab import naive_gemm
from softmax_lab import stable_softmax


def softmax_threads(row, threads=4):
    """Simulate strided lane ownership, then reduce lane-local max/sum."""
    # NaN 的比较恒为 False、Inf 参与减法和除法会得到 NaN/Inf，一旦混进
    # max 或求和就再也无法恢复，所以在入口统一拒绝非有限值。
    if not row or threads <= 0 or not all(map(math.isfinite, row)):
        raise ValueError("finite, nonempty row and positive thread count required")
    # 模拟 GPU 的 stride 访问：lane t 负责下标 t, t+threads, t+2*threads, ...
    # threads 超过行长度时，后面的 lane 分到空列表，对应真实 kernel 里空转的线程。
    ownership = [list(range(t, len(row), threads)) for t in range(threads)]
    # 每 lane 先求局部 max。default=-math.inf 是空 lane 的 identity：
    # 空 lane 绝不能返回 0，否则全负输入的最大值会被 0 抬高（对应故障题 RD02）。
    local_max = [max((row[i] for i in ids), default=-math.inf) for ids in ownership]
    # 两级归约的第一级：合并各 lane 的局部值。真实 GPU 这一层靠 shared memory
    # 写回加 __syncthreads()，或 warp 内 shuffle，这里用一个 max 直接代替。
    row_max = max(local_max)
    # 用全局 row_max 平移后再 exp，所有参数 ≤0，不会上溢（对应故障题 SM01）。
    # fsum 精确求和，减小多元素累加的误差。
    local_sum = [
        math.fsum(math.exp(row[i] - row_max) for i in ids) for ids in ownership
    ]
    # 第二级归约。fsum 让结果与"理想实数求和"一致，因此也掩盖了真实 GPU 上
    # 浮点非结合性带来的、随线程数变化的小差异。
    total = math.fsum(local_sum)
    # 这里每个元素独立重算一次 exp，真实 kernel 会复用已算好的 weights；
    # 数学上等价，只是做了冗余计算（也不模拟任何访存代价）。
    output = [math.exp(x - row_max) / total for x in row]
    # 连同中间状态一起返回，便于测试核对 lane 归属、局部 max/和与最终结果。
    return dict(ownership=ownership, local_max=local_max, row_max=row_max,
                local_sum=local_sum, total=total, output=output)


def welford_state(values):
    """Return (count, mean, M2), M2 is the sum of squared centered differences."""
    n, mean, m2 = 0, 0.0, 0.0
    for value in values:
        # Inf/NaN 一旦进入 mean 或 m2 就永久污染统计量，后续元素无法挽救。
        if not math.isfinite(value):
            raise ValueError("finite values required")
        n += 1
        delta = value - mean
        # 单遍 Welford 递推：均值以 1/n 的步长逼近新样本，m2 累加的是
        # "旧均值偏差 ×（更新后的）新均值偏差"，而不是先把均值定死再回头求平方和。
        # 两遍法在大偏移输入下会发生灾难性抵消，见故障题 LN01 与 RD03。
        mean += delta / n
        m2 += delta * (value - mean)
    return n, mean, m2


def merge_welford(a, b):
    na, ma, m2a = a
    nb, mb, m2b = b
    # 空分区是合并的 identity：直接返回另一半，避免 0/0 或 delta*0/0 产生 NaN。
    if na == 0:
        return b
    if nb == 0:
        return a
    count = na + nb
    # delta 是两个局部均值的差，用来把"组间差异"精确补回 M2。
    delta = mb - ma
    # 合并比"把两段均值再平均一次"稳：均值按各自 count 加权，M2 还要加上
    # delta^2 * na*nb/count 这项组间平方和。结果与分区方式无关（实数意义下），
    # 对应故障题 LN01 的"局部均值不能不带 count 直接平均"。
    return count, ma + delta * nb / count, m2a + m2b + delta * delta * na * nb / count


def gemm_first_tile_trace():
    """T=2 example; only trace the one 2x2 output tile."""
    # A[2,3] @ B[3,2]：K=3 不是 tile(2) 的整数倍，刻意留出 K 方向的尾巴。
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    acc = [[0.0, 0.0], [0.0, 0.0]]
    history = []
    # 两个 K tile：k0=0 覆盖 k=0,1；k0=2 只覆盖 k=2（另一个位置越界）。
    for k0 in (0, 2):
        # 越界的 K 位置补 0，而不是跳过：0 乘任何数还是 0，结果不变，
        # 于是 tile 循环可以写得很规整、不用为尾巴开特例（对应 GM01 的边界装载）。
        at = [[a[r][k0 + t] if k0 + t < 3 else 0.0 for t in range(2)] for r in range(2)]
        bt = [[b[k0 + t][c] if k0 + t < 3 else 0.0 for c in range(2)] for t in range(2)]
        # tile 内三重循环。真实 GPU 用 shared memory 装载 at/bt 再做同样的乘加，
        # 算术完全一样，tile 的意义只在于减少对 global memory 的重复读取。
        for inner in range(2):
            for r in range(2):
                for c in range(2):
                    acc[r][c] += at[r][inner] * bt[inner][c]
        # 记录每个 K tile 之后的累加器快照（r[:] 拷贝，否则后续 in-place 修改会污染历史）。
        history.append(dict(k0=k0, a_tile=at, b_tile=bt, accumulators=[r[:] for r in acc]))
    return history


def blocked_attention(q, k, v, block_size=2, causal=False):
    """Online m/l/o reference for top-left causal attention; no dropout or GQA."""
    if not q or not k or not v or len(k) != len(v):
        raise ValueError("nonempty q/k/v with matching key/value rows required")
    d, dv = len(q[0]), len(v[0])
    if not d or not dv or any(len(row) != d for row in q + k) or any(len(row) != dv for row in v):
        raise ValueError("incompatible or ragged feature dimensions")
    if block_size <= 0:
        raise ValueError("positive block size required")
    if not all(math.isfinite(x) for matrix in (q, k, v) for row in matrix for x in row):
        raise ValueError("finite inputs required")
    scale = 1.0 / math.sqrt(len(q[0]))
    outputs, traces = [], []
    for qi, qrow in enumerate(q):
        # online softmax 的三个运行状态：m 是已见 logit 的最大值，total 是对应
        # 分母 l，acc 是已按权重累加的 o。m=-inf、total=0 让第一个 block 的
        # 重缩放因子自然取 0，无需特判。
        m, total = -math.inf, 0.0
        acc = [0.0] * len(v[0])
        row_trace = []
        for begin in range(0, len(k), block_size):
            # causal 时只保留 j<=qi 的 key；尾部块天然比 block_size 短。
            visible = [j for j in range(begin, min(begin + block_size, len(k)))
                       if not causal or j <= qi]
            # An entirely masked block has no statistics; skip it before -inf - -inf.
            if not visible:
                continue
            # 块内点积用 fsum 精确求和，先算好后统一乘 scale。
            scores = [math.fsum(a * b for a, b in zip(qrow, k[j])) * scale for j in visible]
            block_max = max(scores)
            new_max = max(m, block_max)
            # 重缩放因子：把之前按旧 m 归一化的统计量折算到 new_max 尺度。
            # 首个块 total==0，直接取 0，不依赖 -inf 参与 exp 运算。
            alpha = math.exp(m - new_max) if total else 0.0
            # 块内每项都减 new_max，参数 ≤0，不会上溢。
            weights = [math.exp(s - new_max) for s in scores]
            # 旧分母缩放后与新块分母相加 —— online softmax 的核心，
            # 免去"先扫一遍求全局 max、再扫第二遍"的两遍做法。
            total = alpha * total + math.fsum(weights)
            # 旧的加权和 o 是在旧 max 下算的，max 变了必须乘 alpha 补正，
            # 再叠加本块的贡献。这正是故障题 AT01 的 m/l/o 同步重缩放。
            acc = [alpha * acc[c] + math.fsum(w * v[j][c] for w, j in zip(weights, visible))
                   for c in range(len(acc))]
            m = new_max
            row_trace.append(dict(begin=begin, m=m, l=total, o=acc[:]))
        # 整行被遮罩时 total 仍为 0，继续下去会除零；显式报错。
        if not total:
            raise ValueError("fully masked query row")
        outputs.append([value / total for value in acc])
        traces.append(row_trace)
    return outputs, traces


def im2col_single_channel(x, kernel):
    """Valid stride-1 cross-correlation via explicitly materialized window rows."""
    if not x or not kernel or not x[0] or not kernel[0]:
        raise ValueError("nonempty matrices required")
    h, w, kh, kw = len(x), len(x[0]), len(kernel), len(kernel[0])
    if any(len(row) != w for row in x) or any(len(row) != kw for row in kernel):
        raise ValueError("ragged matrices are unsupported")
    # valid（无 padding）、stride=1 的输出尺寸，和 conv2d_single_channel 的退化情形一致。
    oh, ow = h - kh + 1, w - kw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("kernel larger than input")
    # 把每个输出位置对应的 kh×kw 窗口拍平成一行，得到 (oh*ow, kh*kw) 的矩阵。
    # 这正是 implicit GEMM 的思路：卷积变成一次矩阵乘。真实 GPU 不会真的物化
    # 这个矩阵（内存放大 kh*kw 倍），而是用索引现算——这里为了可见性才显式展开。
    windows = [
        [x[oy + ky][ox + kx] for ky in range(kh) for kx in range(kw)]
        for oy in range(oh) for ox in range(ow)
    ]
    # kernel 拍平成列向量。拍平顺序必须与 window 完全一致（都是 ky 外层、kx 内层），
    # 否则会拿权重的第 k 项去乘窗口的第 k 项，结果错而不报错。
    weights = [[value] for row in kernel for value in row]
    product = naive_gemm(windows, weights)
    # 把 (oh*ow, 1) 折回 (oh, ow)。
    return [[product[y * ow + col][0] for col in range(ow)] for y in range(oh)], windows


def show(topic):
    if topic in ("all", "softmax"):
        # 刻意用 1000 附近的数：这正是 SM01 里"普通负数通过、极负值 NaN"的输入风格，
        # 不加平移就会在 exp 里溢出。
        row = [1000.0, 1001.0, 999.0, 998.0, 1002.0]
        state = softmax_threads(row)
        print("SOFTMAX", state)
        print("reference:", stable_softmax(row))
    if topic in ("all", "welford"):
        # 把同一个序列切成两段各自统计再合并，用来核对分区不变性（LN01）。
        left, right = welford_state([1.0, 2.0]), welford_state([3.0, 4.0])
        merged = merge_welford(left, right)
        print("WELFORD left/right/merged:", left, right, merged)
        # M2/count 即总体方差（分母用 count，不是 count-1）。
        print("population variance:", merged[2] / merged[0])
    if topic in ("all", "gemm"):
        for state in gemm_first_tile_trace():
            print("GEMM", state)
    if topic in ("all", "attention"):
        # block_size=1、K 只有两行：每一步都会触发新的 max 与重缩放。
        output, trace = blocked_attention([[1.0]], [[0.0], [2.0]], [[10.0], [20.0]], 1)
        print("ATTENTION trace:", trace)
        print("output:", output)
    if topic in ("all", "conv"):
        output, windows = im2col_single_channel(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 0], [0, -1]])
        print("IM2COL windows:", windows)
        print("output:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", choices=["all", "softmax", "welford", "gemm", "attention", "conv"],
                        default="all")
    show(parser.parse_args().topic)
