"""可运行的进阶算子 reference；这里只验证语义，不模拟 GPU 性能。"""

import math


def attention(q, k, v, causal=False):
    if not q or not k or len(k) != len(v):
        raise ValueError("q/k/v must be non-empty and k/v rows must match")
    d = len(q[0])
    dv = len(v[0])
    if d == 0 or any(len(row) != d for row in q + k):
        raise ValueError("q/k feature dimensions must match")
    if dv == 0 or any(len(row) != dv for row in v):
        raise ValueError("v rows must have equal nonzero width")
    scale = 1.0 / math.sqrt(d)
    output = []
    for qi, qrow in enumerate(q):
        logits = []
        for kj, krow in enumerate(k):
            score = sum(a * b for a, b in zip(qrow, krow)) * scale
            logits.append(score if not causal or kj <= qi else -math.inf)
        m = max(logits)
        if m == -math.inf:
            raise ValueError("fully masked row requires an explicit policy")
        weights = [math.exp(score - m) for score in logits]
        total = sum(weights)
        output.append([
            sum(weights[j] * v[j][col] for j in range(len(k))) / total
            for col in range(dv)
        ])
    return output


def conv2d_single_channel(x, kernel, stride=1, pad=0):
    if stride <= 0 or pad < 0 or not x or not kernel:
        raise ValueError("invalid shape, stride, or padding")
    h, w = len(x), len(x[0])
    kh, kw = len(kernel), len(kernel[0])
    if not w or not kw or any(len(row) != w for row in x) or any(len(row) != kw for row in kernel):
        raise ValueError("ragged arrays are unsupported")
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("kernel larger than padded input")
    output = []
    for oy in range(oh):
        row = []
        for ox in range(ow):
            acc = 0.0
            for ky in range(kh):
                iy = oy * stride + ky - pad
                for kx in range(kw):
                    ix = ox * stride + kx - pad
                    if 0 <= iy < h and 0 <= ix < w:
                        acc += x[iy][ix] * kernel[ky][kx]
            row.append(acc)
        output.append(row)
    return output


def bias_relu(x, bias):
    if not x or not bias or any(len(row) != len(bias) for row in x):
        raise ValueError("x columns and bias width must match")
    return [[max(value + bias[col], 0.0) for col, value in enumerate(row)] for row in x]


if __name__ == "__main__":
    print("attention:", attention([[1.0], [0.0]], [[1.0], [0.0]], [[10.0], [20.0]]))
    print("causal attention:", attention([[1.0], [0.0]], [[1.0], [0.0]], [[10.0], [20.0]], causal=True))
    print("conv2d:", conv2d_single_channel([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 0], [0, -1]]))
    print("bias+relu:", bias_relu([[-2.0, 1.0], [3.0, -4.0]], [1.0, 2.0]))
