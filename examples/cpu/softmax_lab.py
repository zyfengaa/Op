"""稳定 Softmax 和逐块在线归一化；这里用 Python 双精度演示公式。"""

from math import exp, fsum


def stable_softmax(x: list[float]) -> list[float]:
    if not x:
        raise ValueError("softmax needs a non-empty row")
    m = max(x)
    weights = [exp(value - m) for value in x]
    denominator = fsum(weights)
    return [weight / denominator for weight in weights]


def online_normalizer(x: list[float], block_size: int) -> tuple[float, float]:
    """返回 (m, l)，其中 l = sum(exp(x - m))。"""
    if not x or block_size <= 0:
        raise ValueError("non-empty x and positive block_size required")
    m, l = float("-inf"), 0.0
    for begin in range(0, len(x), block_size):
        block = x[begin:begin + block_size]
        block_m = max(block)
        block_l = fsum(exp(value - block_m) for value in block)
        new_m = max(m, block_m)
        l = l * exp(m - new_m) + block_l * exp(block_m - new_m)
        m = new_m
    return m, l


def main() -> None:
    x = [1000.0, 1001.0, 999.0, 998.0, 997.0]
    probs = stable_softmax(x)
    m, l = online_normalizer(x, block_size=2)
    online = [exp(value - m) / l for value in x]
    print("probabilities:", [round(p, 6) for p in probs])
    print("sum:", round(fsum(probs), 6))
    print("online equals stable:", all(abs(a - b) < 1e-12 for a, b in zip(probs, online)))


if __name__ == "__main__":
    main()
