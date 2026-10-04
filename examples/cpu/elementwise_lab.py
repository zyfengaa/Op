"""三个逐元素操作，演示公式、边界和值域。Python 3.10+。"""

from math import erf, exp, sqrt


def vector_add(a: list[float], b: list[float]) -> list[float]:
    if len(a) != len(b):
        raise ValueError("A and B must have equal length")
    return [left + right for left, right in zip(a, b)]


def relu(x: list[float]) -> list[float]:
    return [max(value, 0.0) for value in x]


def sigmoid_scalar(x: float) -> float:
    # 分支避免对极大正数计算 exp(x) 而溢出。
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def gelu_exact(x: float) -> float:
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def main() -> None:
    a, b = [1.0, -2.0, 3.0], [4.0, 5.0, -6.0]
    print("add:", vector_add(a, b))
    print("relu:", relu(a))
    print("sigmoid(1000):", sigmoid_scalar(1000.0))
    print("sigmoid(-1000):", sigmoid_scalar(-1000.0))
    print("gelu_exact(0):", gelu_exact(0.0))


if __name__ == "__main__":
    main()
