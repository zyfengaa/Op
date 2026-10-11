"""可运行的进阶算子 reference；这里只验证语义，不模拟 GPU 性能。"""

import math


def attention(q, k, v, causal=False):
    # K 与 V 的行必须一一对应：每个 key 配一个 value。行数不等时下面的 zip 会
    # 静默截断，算出的是"被砍掉尾巴"的注意力，不报错但结果错。
    if not q or not k or len(k) != len(v):
        raise ValueError("q/k/v must be non-empty and k/v rows must match")
    # d 是 query/key 的特征维，dv 是 value 的输出维；两者可以不同，不能混用。
    d = len(q[0])
    dv = len(v[0])
    if d == 0 or any(len(row) != d for row in q + k):
        raise ValueError("q/k feature dimensions must match")
    if dv == 0 or any(len(row) != dv for row in v):
        raise ValueError("v rows must have equal nonzero width")
    # 1/sqrt(d)：点积的方差随 d 增长，不缩放会让 softmax 落进饱和区、梯度趋近 0。
    scale = 1.0 / math.sqrt(d)
    output = []
    for qi, qrow in enumerate(q):
        logits = []
        for kj, krow in enumerate(k):
            # 用 sum（不是 fsum）且靠 zip：上面已校验同宽，所以不会截断；
            # 累加顺序即 kj 递增，与真实 kernel 的 K 循环方向一致。
            score = sum(a * b for a, b in zip(qrow, krow)) * scale
            # 因果遮罩用 -inf 而不是 0：exp(-inf) 恒为 0，未来位置权重严格为零。
            # 若改成 0，未来位置会拿到 exp(0)=1 的权重，遮罩形同虚设。
            logits.append(score if not causal or kj <= qi else -math.inf)
        # 先减行最大值再 exp：参数全部 ≤0，结果落在 (0,1]，不会上溢。
        m = max(logits)
        # 整行都被遮罩时没有可归一的权重，必须显式处理，否则分母为 0 得到 NaN。
        if m == -math.inf:
            raise ValueError("fully masked row requires an explicit policy")
        weights = [math.exp(score - m) for score in logits]
        # 至少有一个 score==m，所以 total>=1，不会下溢到 0 而触发除零。
        total = sum(weights)
        output.append([
            sum(weights[j] * v[j][col] for j in range(len(k))) / total
            for col in range(dv)
        ])
    return output


def conv2d_single_channel(x, kernel, stride=1, pad=0):
    # stride=0 会让下面的输出尺寸推导变成除零；pad<0 会把窗口挪到数组外。
    if stride <= 0 or pad < 0 or not x or not kernel:
        raise ValueError("invalid shape, stride, or padding")
    h, w = len(x), len(x[0])
    kh, kw = len(kernel), len(kernel[0])
    if not w or not kw or any(len(row) != w for row in x) or any(len(row) != kw for row in kernel):
        raise ValueError("ragged arrays are unsupported")
    # 输出尺寸公式，整数除法向下取整：右/下侧不足一个完整窗口的输入被丢弃，
    # 这正是 valid 卷积的边界语义。
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("kernel larger than padded input")
    # 这是 cross-correlation（不翻转 kernel），深度学习里的 "conv2d" 都是这个定义。
    output = []
    for oy in range(oh):
        row = []
        for ox in range(ow):
            acc = 0.0
            for ky in range(kh):
                iy = oy * stride + ky - pad
                for kx in range(kw):
                    ix = ox * stride + kx - pad
                    # 越界即视作 0：padding 不真的补一块内存，而是靠这里跳过。
                    # 少了这个判断就会读到相邻行/列的数据，得到错误的边界。
                    if 0 <= iy < h and 0 <= ix < w:
                        acc += x[iy][ix] * kernel[ky][kx]
            row.append(acc)
        output.append(row)
    return output


def bias_relu(x, bias):
    # bias 按列广播。某行宽度与 bias 不等时 enumerate 会错位或漏掉通道，
    # 所以统一要求每行等宽（对应故障题 VA03 的广播错位症状）。
    if not x or not bias or any(len(row) != len(bias) for row in x):
        raise ValueError("x columns and bias width must match")
    # 先加 bias 再 relu：这是 epilogue 融合，省掉中间结果的一次写出与重读。
    return [[max(value + bias[col], 0.0) for col, value in enumerate(row)] for row in x]


if __name__ == "__main__":
    # 固定输入、固定输出，仅作 smoke test；不比较 GPU 结果，也不测性能。
    print("attention:", attention([[1.0], [0.0]], [[1.0], [0.0]], [[10.0], [20.0]]))
    print("causal attention:", attention([[1.0], [0.0]], [[1.0], [0.0]], [[10.0], [20.0]], causal=True))
    print("conv2d:", conv2d_single_channel([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 0], [0, -1]]))
    print("bias+relu:", bias_relu([[-2.0, 1.0], [3.0, -4.0]], [1.0, 2.0]))
