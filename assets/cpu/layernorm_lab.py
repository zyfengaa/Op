"""行 LayerNorm：先求均值，再求方差和仿射变换。"""

from math import fsum, sqrt


def layer_norm(
    row: list[float], gamma: list[float], beta: list[float], eps: float = 1e-5
) -> list[float]:
    if not row or len(row) != len(gamma) or len(row) != len(beta):
        raise ValueError("row/gamma/beta must be non-empty and equal length")
    if eps <= 0:
        raise ValueError("eps must be positive")
    mean = fsum(row) / len(row)
    variance = fsum((value - mean) ** 2 for value in row) / len(row)
    inv_std = 1.0 / sqrt(variance + eps)
    return [(value - mean) * inv_std * scale + bias
            for value, scale, bias in zip(row, gamma, beta)]


def main() -> None:
    row = [1.0, 2.0, 3.0, 4.0]
    gamma = [1.0] * 4
    beta = [0.0] * 4
    print("layernorm:", [round(v, 6) for v in layer_norm(row, gamma, beta)])
    constant = [7.0] * 4
    print("constant row:", layer_norm(constant, gamma, beta))


if __name__ == "__main__":
    main()
