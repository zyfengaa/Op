"""零依赖的 GEMM 学习程序。

它不是 GPU benchmark，而是帮助初学者理解 naive/tiled GEMM 的等价性、
边界处理和 reference 校验。真正的 GPU 性能需要在 CUDA/Triton 等后端上测量。
"""

from math import isclose


def make_matrix(rows: int, cols: int) -> list[list[float]]:
    # 用 (r*cols + c) % 7 - 3 生成随下标变化的周期值，而不是全 1 或纯随机数：
    # 常数矩阵会让"读错下标"和"读对下标"算出相同结果，把索引类 bug 藏住；
    # 纯随机数又让人无法手算核对。可复现的周期值兼顾两者。
    return [[float((r * cols + c) % 7 - 3) for c in range(cols)] for r in range(rows)]


def naive_gemm(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    # k 同时是 A 的列数、B 的行数、内层累加长度；三者必须一致，矩阵乘法才有定义。
    m, k = len(a), len(a[0])
    # 在这里显式拒绝不匹配的维度，避免下面 range(k) 去索引 b 时越界或静默算错。
    if len(b) != k:
        raise ValueError("A 的列数必须等于 B 的行数")
    n = len(b[0])
    # 三重循环是矩阵乘法定义的直译：每个 C[row][col] 是 A 第 row 行与 B 第 col 列的点积。
    # 它是后面 tiled 版本做逐元素对拍的基准。
    return [[sum(a[row][inner] * b[inner][col] for inner in range(k)) for col in range(n)] for row in range(m)]


def tiled_gemm(a: list[list[float]], b: list[list[float]], tile: int) -> list[list[float]]:
    m, k = len(a), len(a[0])
    n = len(b[0])
    # 先清零再累加：分块把内层循环切成多段，每段都必须累加到已有结果上，不能每段覆盖写。
    out = [[0.0 for _ in range(n)] for _ in range(m)]
    # 这里用列表模拟 tile；GPU 版本对应 shared-memory tile。
    for row0 in range(0, m, tile):
        for col0 in range(0, n, tile):
            for inner0 in range(0, k, tile):
                # 三处 min(...) 是边界处理：最后一个 tile 通常取不满，
                # 用 min 把上界夹回矩阵实际尺寸，等价于 CUDA 里对尾块的边界判断。
                # 分块的意义在于同一份 A/B 数据能被多个输出元素复用——GPU 上正是把它搬进
                # shared memory 反复消费，减少对 global memory 的重复读取。
                for row in range(row0, min(row0 + tile, m)):
                    for col in range(col0, min(col0 + tile, n)):
                        for inner in range(inner0, min(inner0 + tile, k)):
                            out[row][col] += a[row][inner] * b[inner][col]
    return out


def assert_same(expected: list[list[float]], actual: list[list[float]]) -> None:
    # 用 zip 成对比较而非按维索引：形状不一致时 zip 会静默截断到较短的一边。
    # 教学例的形状由构造保证，所以这里可接受；真实对拍应先断言 shape 相等再逐元素比。
    for expected_row, actual_row in zip(expected, actual):
        for expected_value, actual_value in zip(expected_row, actual_row):
            # 浮点加法不满足结合律，分块改变了累加顺序，会带来舍入差异，
            # 所以用 rel/abs 容差比较而不是 ==；容差大小本身也该写进结论。
            if not isclose(expected_value, actual_value, rel_tol=1e-6, abs_tol=1e-6):
                raise AssertionError((expected_value, actual_value))


def main() -> None:
    # 特意使用不能被 tile=3 整除的尺寸，验证边界逻辑。
    m, k, n, tile = 5, 7, 4, 3
    a = make_matrix(m, k)
    b = make_matrix(k, n)
    reference = naive_gemm(a, b)
    tiled = tiled_gemm(a, b, tile)
    # tiled 必须与 naive 逐元素一致：分块只改变计算顺序和访存模式，不改变数学结果。
    # 二者不符说明 tile 的索引或边界写错了。
    assert_same(reference, tiled)
    print("GEMM correctness: PASS")
    print(f"shape: A[{m},{k}] x B[{k},{n}] -> C[{m},{n}]")
    print(f"tile: {tile} (shape intentionally not divisible by tile)")
    # FLOPs 是理论计算量计数（每个输出元素 k 次乘加、每次记 2 FLOP），不是耗时、更不是性能。
    # 打印它只为建立量级概念，不能据此声称速度。
    print(f"FLOPs: {2 * m * n * k}")
    print("下一步：把 tiled_gemm 的 tile 搬运过程映射到 CUDA shared memory，再用 profiler 验证性能假设。")


if __name__ == "__main__":
    main()
