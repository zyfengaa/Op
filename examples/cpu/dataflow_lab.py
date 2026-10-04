"""Print intermediate states that connect mathematics to kernel dataflow.

Run from the repo root: python examples/cpu/dataflow_lab.py
Finite Python floats only; timings are deliberately not presented as GPU evidence.
"""

import argparse
import math

from gemm_lab import naive_gemm
from softmax_lab import stable_softmax


def softmax_threads(row, threads=4):
    """Simulate strided lane ownership, then reduce lane-local max/sum."""
    if not row or threads <= 0 or not all(map(math.isfinite, row)):
        raise ValueError("finite, nonempty row and positive thread count required")
    ownership = [list(range(t, len(row), threads)) for t in range(threads)]
    local_max = [max((row[i] for i in ids), default=-math.inf) for ids in ownership]
    row_max = max(local_max)
    local_sum = [
        math.fsum(math.exp(row[i] - row_max) for i in ids) for ids in ownership
    ]
    total = math.fsum(local_sum)
    output = [math.exp(x - row_max) / total for x in row]
    return dict(ownership=ownership, local_max=local_max, row_max=row_max,
                local_sum=local_sum, total=total, output=output)


def welford_state(values):
    """Return (count, mean, M2), M2 is the sum of squared centered differences."""
    n, mean, m2 = 0, 0.0, 0.0
    for value in values:
        if not math.isfinite(value):
            raise ValueError("finite values required")
        n += 1
        delta = value - mean
        mean += delta / n
        m2 += delta * (value - mean)
    return n, mean, m2


def merge_welford(a, b):
    na, ma, m2a = a
    nb, mb, m2b = b
    if na == 0:
        return b
    if nb == 0:
        return a
    count = na + nb
    delta = mb - ma
    return count, ma + delta * nb / count, m2a + m2b + delta * delta * na * nb / count


def gemm_first_tile_trace():
    """T=2 example; only trace the one 2x2 output tile."""
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    acc = [[0.0, 0.0], [0.0, 0.0]]
    history = []
    for k0 in (0, 2):
        at = [[a[r][k0 + t] if k0 + t < 3 else 0.0 for t in range(2)] for r in range(2)]
        bt = [[b[k0 + t][c] if k0 + t < 3 else 0.0 for c in range(2)] for t in range(2)]
        for inner in range(2):
            for r in range(2):
                for c in range(2):
                    acc[r][c] += at[r][inner] * bt[inner][c]
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
        m, total = -math.inf, 0.0
        acc = [0.0] * len(v[0])
        row_trace = []
        for begin in range(0, len(k), block_size):
            visible = [j for j in range(begin, min(begin + block_size, len(k)))
                       if not causal or j <= qi]
            # An entirely masked block has no statistics; skip it before -inf - -inf.
            if not visible:
                continue
            scores = [math.fsum(a * b for a, b in zip(qrow, k[j])) * scale for j in visible]
            block_max = max(scores)
            new_max = max(m, block_max)
            alpha = math.exp(m - new_max) if total else 0.0
            weights = [math.exp(s - new_max) for s in scores]
            total = alpha * total + math.fsum(weights)
            acc = [alpha * acc[c] + math.fsum(w * v[j][c] for w, j in zip(weights, visible))
                   for c in range(len(acc))]
            m = new_max
            row_trace.append(dict(begin=begin, m=m, l=total, o=acc[:]))
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
    oh, ow = h - kh + 1, w - kw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("kernel larger than input")
    windows = [
        [x[oy + ky][ox + kx] for ky in range(kh) for kx in range(kw)]
        for oy in range(oh) for ox in range(ow)
    ]
    weights = [[value] for row in kernel for value in row]
    product = naive_gemm(windows, weights)
    return [[product[y * ow + col][0] for col in range(ow)] for y in range(oh)], windows


def show(topic):
    if topic in ("all", "softmax"):
        row = [1000.0, 1001.0, 999.0, 998.0, 1002.0]
        state = softmax_threads(row)
        print("SOFTMAX", state)
        print("reference:", stable_softmax(row))
    if topic in ("all", "welford"):
        left, right = welford_state([1.0, 2.0]), welford_state([3.0, 4.0])
        merged = merge_welford(left, right)
        print("WELFORD left/right/merged:", left, right, merged)
        print("population variance:", merged[2] / merged[0])
    if topic in ("all", "gemm"):
        for state in gemm_first_tile_trace():
            print("GEMM", state)
    if topic in ("all", "attention"):
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
