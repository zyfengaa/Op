"""三个逐元素操作，演示公式、边界和值域。Python 3.10+。"""

from math import erf, exp, sqrt


def vector_add(a: list[float], b: list[float]) -> list[float]:
    # zip 会静默截断到较短的那个，长度不等时必须显式报错：
    # 否则尾部元素会无声消失，和"只算了一截"的故障症状同类。
    if len(a) != len(b):
        raise ValueError("A and B must have equal length")
    # 空输入返回空列表是合法边界（len 相等即可），不特判。
    return [left + right for left, right in zip(a, b)]


def relu(x: list[float]) -> list[float]:
    # max(value, 0.0) 就是阈值；注意 NaN 的传播取决于参数顺序，不要依赖它。
    return [max(value, 0.0) for value in x]


def sigmoid_scalar(x: float) -> float:
    # 分支避免对极大正数计算 exp(x) 而溢出。
    # 两个分支数学上恒等：把分子分母同乘 exp(-|x|) 即可互推。
    # 差别全在浮点上——负号保证 exp 的指数永远 ≤0，永不溢出。
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    # x<0 时 exp(x) 会下溢到 0，得到 0/(1+0)=0.0，正是 sigmoid 的极值，
    # 因此 sigmoid(-1000) 精确等于 0.0 而不是一个很小的数。这是 SM01 里
    # "平移等价性"思想在逐元素算子上的体现。
    z = exp(x)
    return z / (1.0 + z)


def gelu_exact(x: float) -> float:
    # erf 版本是精确 GELU。x 很大时 erf→1，gelu≈x；x 极负时 erf→-1，gelu≈0。
    # 生产实现常用 tanh 近似以避免 erf 的开销，这里保留精确式便于对拍。
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def main() -> None:
    # ±1000 专门用来验证上面分支的边界：结果应当是精确的 1.0 与 0.0。
    a, b = [1.0, -2.0, 3.0], [4.0, 5.0, -6.0]
    print("add:", vector_add(a, b))
    print("relu:", relu(a))
    print("sigmoid(1000):", sigmoid_scalar(1000.0))
    print("sigmoid(-1000):", sigmoid_scalar(-1000.0))
    print("gelu_exact(0):", gelu_exact(0.0))


if __name__ == "__main__":
    main()
