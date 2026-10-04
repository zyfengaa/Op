"""零依赖的 GEMM 学习程序。

它不是 GPU benchmark，而是帮助初学者理解 naive/tiled GEMM 的等价性、
边界处理和 reference 校验。真正的 GPU 性能需要在 CUDA/Triton 等后端上测量。
"""

from math import isclose


def make_matrix(rows: int, cols: int) -> list[list[float]]:
    return [[float((r * cols + c) % 7 - 3) for c in range(cols)] for r in range(rows)]


def naive_gemm(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    m, k = len(a), len(a[0])
    if len(b) != k:
        raise ValueError("A 的列数必须等于 B 的行数")
    n = len(b[0])
    return [[sum(a[row][inner] * b[inner][col] for inner in range(k)) for col in range(n)] for row in range(m)]


def tiled_gemm(a: list[list[float]], b: list[list[float]], tile: int) -> list[list[float]]:
    m, k = len(a), len(a[0])
    n = len(b[0])
    out = [[0.0 for _ in range(n)] for _ in range(m)]
    # 这里用列表模拟 tile；GPU 版本对应 shared-memory tile。
    for row0 in range(0, m, tile):
        for col0 in range(0, n, tile):
            for inner0 in range(0, k, tile):
                for row in range(row0, min(row0 + tile, m)):
                    for col in range(col0, min(col0 + tile, n)):
                        for inner in range(inner0, min(inner0 + tile, k)):
                            out[row][col] += a[row][inner] * b[inner][col]
    return out


def assert_same(expected: list[list[float]], actual: list[list[float]]) -> None:
    for expected_row, actual_row in zip(expected, actual):
        for expected_value, actual_value in zip(expected_row, actual_row):
            if not isclose(expected_value, actual_value, rel_tol=1e-6, abs_tol=1e-6):
                raise AssertionError((expected_value, actual_value))


def main() -> None:
    # 特意使用不能被 tile=3 整除的尺寸，验证边界逻辑。
    m, k, n, tile = 5, 7, 4, 3
    a = make_matrix(m, k)
    b = make_matrix(k, n)
    reference = naive_gemm(a, b)
    tiled = tiled_gemm(a, b, tile)
    assert_same(reference, tiled)
    print("GEMM correctness: PASS")
    print(f"shape: A[{m},{k}] x B[{k},{n}] -> C[{m},{n}]")
    print(f"tile: {tile} (shape intentionally not divisible by tile)")
    print(f"FLOPs: {2 * m * n * k}")
    print("下一步：把 tiled_gemm 的 tile 搬运过程映射到 CUDA shared memory，再用 profiler 验证性能假设。")


if __name__ == "__main__":
    main()
