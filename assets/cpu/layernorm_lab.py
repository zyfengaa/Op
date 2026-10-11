"""行 LayerNorm：先求均值，再求方差和仿射变换。"""

from math import fsum, sqrt


def layer_norm(
    row: list[float], gamma: list[float], beta: list[float], eps: float = 1e-5
) -> list[float]:
    # 三者必须同宽：宽度不等时 zip 会静默截断，部分通道既不被归一化也不被缩放。
    # 空行还会让下面除以 0。LayerNorm 逐行独立，与 batch 无关。
    if not row or len(row) != len(gamma) or len(row) != len(beta):
        raise ValueError("row/gamma/beta must be non-empty and equal length")
    # eps 是常数行（方差为 0）的保护；≤0 时常数行会得到 1/sqrt(0)=inf 或 sqrt(负数)。
    if eps <= 0:
        raise ValueError("eps must be positive")
    # fsum 精确求和，避免长行上朴素累加的漂移；除以 len(row) 得到行均值。
    mean = fsum(row) / len(row)
    # 两遍法：先定死均值，再累加平方偏差。对 10000 附近的输入这种写法会有
    # 灾难性抵消；工业实现多用 Welford 单遍或先减一个粗均值（见 LN01/RD03）。
    variance = fsum((value - mean) ** 2 for value in row) / len(row)
    # 先算倒数再逐元素相乘，比每个元素各除一次便宜。
    # eps 加在方差上、开方之前，而不是加在标准差上，两者数值上不同。
    inv_std = 1.0 / sqrt(variance + eps)
    # 顺序是"先归一化、再仿射"：γ 缩放、β 平移。
    return [(value - mean) * inv_std * scale + bias
            for value, scale, bias in zip(row, gamma, beta)]


def main() -> None:
    row = [1.0, 2.0, 3.0, 4.0]
    gamma = [1.0] * 4
    beta = [0.0] * 4
    print("layernorm:", [round(v, 6) for v in layer_norm(row, gamma, beta)])
    # 常数行的方差为 0，是 eps 保护的关键用例：期望输出全 0（因为 value-mean=0），
    # 而不是 inf 或 NaN。若 eps 处理不当，这里立刻暴露。
    constant = [7.0] * 4
    print("constant row:", layer_norm(constant, gamma, beta))


if __name__ == "__main__":
    main()
