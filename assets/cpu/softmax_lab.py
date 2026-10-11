"""稳定 Softmax 和逐块在线归一化；这里用 Python 双精度演示公式。"""

from math import exp, fsum


def stable_softmax(x: list[float]) -> list[float]:
    # 空行没有 max，分母也会是 0。直接报错，不给"返回空"这种含糊语义。
    if not x:
        raise ValueError("softmax needs a non-empty row")
    # 先减行最大值再 exp：参数全部 ≤0，结果落在 (0,1]，绝不会上溢。
    # 这就是 SM01 的解：不加平移时，logit≈1000 会让 exp 溢出成 inf，
    # inf/inf 得 NaN；而 softmax(x) == softmax(x-m) 是恒等式，平移不改变数学结果，
    # 只改变浮点行为（平移等价性）。
    m = max(x)
    weights = [exp(value - m) for value in x]
    # 至少有一个 value==m，得到 exp(0)=1，所以分母 ≥1，不会下溢到 0 而触发除零。
    # fsum 精确求和，减小累加误差。
    denominator = fsum(weights)
    return [weight / denominator for weight in weights]


def online_normalizer(x: list[float], block_size: int) -> tuple[float, float]:
    """返回 (m, l)，其中 l = sum(exp(x - m))。"""
    if not x or block_size <= 0:
        raise ValueError("non-empty x and positive block_size required")
    # 初始状态：还没见过任何元素，max 是 -inf、分母是 0，这样第一个 block 的
    # 合并公式自然成立。绝不能把 m 初始化为 0 —— 全负输入会因此得到错误的 max
    # （与 RD02 的"错误 identity"是同一类错误）。
    m, l = float("-inf"), 0.0
    for begin in range(0, len(x), block_size):
        block = x[begin:begin + block_size]
        # 块内先各自用块内 max 归一化，保证块内不溢出。
        block_m = max(block)
        block_l = fsum(exp(value - block_m) for value in block)
        new_m = max(m, block_m)
        # 把"旧尺度下的分母"和"本块尺度下的分母"都折算到 new_m 尺度再相加：
        # exp(m-new_m) 给旧项、exp(block_m-new_m) 给本块。当 new_m 恰等于其中一项时，
        # 对应因子为 exp(0)=1。这是 online softmax 的关键（对应 AT01 的 m/l/o 重缩放）——
        # 只需单遍扫描，不必先求全局 max。
        l = l * exp(m - new_m) + block_l * exp(block_m - new_m)
        m = new_m
    # 注意只返回 (m, l)，指数项没被保存；调用方需自己用 exp(value-m)/l 还原。
    return m, l


def main() -> None:
    # 1000 附近的数专门用来暴露下溢/溢出：不加减 max 的版本会直接 NaN（SM01）。
    x = [1000.0, 1001.0, 999.0, 998.0, 997.0]
    probs = stable_softmax(x)
    m, l = online_normalizer(x, block_size=2)
    online = [exp(value - m) / l for value in x]
    print("probabilities:", [round(p, 6) for p in probs])
    print("sum:", round(fsum(probs), 6))
    # 分块单遍的结果必须与两遍的 stable_softmax 一致到极高精度。
    print("online equals stable:", all(abs(a - b) < 1e-12 for a, b in zip(probs, online)))


if __name__ == "__main__":
    main()
