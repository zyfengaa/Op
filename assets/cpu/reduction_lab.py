"""模拟 block 内树形归约：identity、padding 和每轮状态。Python 3.10+。"""


def tree_sum(values: list[float]) -> tuple[float, list[list[float]]]:
    if not values:
        return 0.0, []
    width = 1 << (len(values) - 1).bit_length()
    partial = values[:] + [0.0] * (width - len(values))
    history = [partial[:]]
    offset = width // 2
    while offset:
        for i in range(offset):
            partial[i] += partial[i + offset]
        history.append(partial[:])
        offset //= 2
    return partial[0], history


def tree_max(values: list[float]) -> float:
    if not values:
        raise ValueError("max of empty input is undefined")
    width = 1 << (len(values) - 1).bit_length()
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
    print(f"input={values}, padded_width={len(history[0])}")
    for step, partial in enumerate(history):
        print(f"step {step}: {partial}")
    print("sum:", total)
    print("max([-5,-2,-9]):", tree_max([-5.0, -2.0, -9.0]))
    print("mean:", total / len(values))


if __name__ == "__main__":
    main()
