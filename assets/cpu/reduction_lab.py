"""模拟 block 内树形归约：identity、padding 和每轮状态。Python 3.10+。"""


def tree_sum(values: list[float]) -> tuple[float, list[list[float]]]:
    # 空输入的 sum 有定义，返回加法 identity 0.0；history 为空列表而不是报错。
    if not values:
        return 0.0, []
    # 向上取到 2 的幂：树形归约每轮把宽度砍半，非 2 的幂长度会在某一轮出现
    # 没有搭档的尾部元素而被漏掉（对应故障题 RD04）。
    width = 1 << (len(values) - 1).bit_length()
    # 用 0.0 补齐恰好是加法的 identity，补进去不改变结果。选错 identity
    # （例如给 max 归约补 0）就会算错 —— 见 RD02。这里的补齐是"数学上的无害"，
    # 不同于 RD03 里"把 padding 数量也算进分母"的错误。
    partial = values[:] + [0.0] * (width - len(values))
    # 记录第 0 轮（补齐后）的快照；[:] 拷贝，否则后面的 in-place 修改会连历史一起改。
    history = [partial[:]]
    # offset 从 width//2 开始逐轮减半：第 i 个位置吸收第 i+offset 个。
    # 这个顺序就是 GPU 上 shared memory 树形归约 / warp shuffle 下降的模式。
    offset = width // 2
    while offset:
        for i in range(offset):
            # in-place 累加；后半段本轮被读完，下一轮宽度减半就不再被读，不会重复计数。
            partial[i] += partial[i + offset]
        history.append(partial[:])
        offset //= 2
    # 返回根节点（下标 0）上的总和。
    return partial[0], history


def tree_max(values: list[float]) -> float:
    # max 的空输入没有定义，不能像 sum 那样返回一个"中性值"：任何有限值都可能是
    # 真实结果，而 -inf 又超出元素的取值范围。只能显式报错。
    if not values:
        raise ValueError("max of empty input is undefined")
    width = 1 << (len(values) - 1).bit_length()
    # max 的 identity 是 -inf，不是 0。补 0 会让全负数输入的最大值变成 0 ——
    # 这正是故障题 RD02 的症状。所以这里必须补 -inf。
    partial = values[:] + [float("-inf")] * (width - len(values))
    offset = width // 2
    while offset:
        for i in range(offset):
            partial[i] = max(partial[i], partial[i + offset])
        offset //= 2
    return partial[0]


def main() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    total, history = tree_sum(values)
    # len(history[0]) 是补齐后的宽度（8），不是真实长度（5）。
    print(f"input={values}, padded_width={len(history[0])}")
    for step, partial in enumerate(history):
        print(f"step {step}: {partial}")
    print("sum:", total)
    # 全负输入用来看 max 的 identity：如果 padding 补 0 就会得到 0 而不是 -2。
    print("max([-5,-2,-9]):", tree_max([-5.0, -2.0, -9.0]))
    # 均值必须除以真实元素个数 len(values)，不是补齐宽度 width ——
    # 用 width 会让 5 个数的均值偏小，即 RD03 的症状。
    print("mean:", total / len(values))


if __name__ == "__main__":
    main()
