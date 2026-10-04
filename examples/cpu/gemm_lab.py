"""GEMM 手算实验：naive 与 tiled、非整除尺寸、tile load 次数。"""


def validate(a: list[list[float]], b: list[list[float]]) -> tuple[int, int, int]:
    if not a or not b or not a[0] or not b[0]:
        raise ValueError("non-empty matrices required")
    m, k, n = len(a), len(a[0]), len(b[0])
    if any(len(row) != k for row in a) or len(b) != k or any(len(row) != n for row in b):
        raise ValueError("incompatible or ragged matrices")
    return m, n, k


def naive_gemm(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    m, n, k = validate(a, b)
    return [[sum(a[row][inner] * b[inner][col] for inner in range(k))
             for col in range(n)] for row in range(m)]


def tiled_gemm(
    a: list[list[float]], b: list[list[float]], tile: int
) -> tuple[list[list[float]], int]:
    m, n, k = validate(a, b)
    if tile <= 0:
        raise ValueError("tile must be positive")
    c = [[0.0] * n for _ in range(m)]
    tile_loads = 0
    for row0 in range(0, m, tile):
        for col0 in range(0, n, tile):
            for k0 in range(0, k, tile):
                a_tile = [[0.0] * tile for _ in range(tile)]
                b_tile = [[0.0] * tile for _ in range(tile)]
                for i in range(tile):
                    for j in range(tile):
                        if row0 + i < m and k0 + j < k:
                            a_tile[i][j] = a[row0 + i][k0 + j]
                            tile_loads += 1
                        if k0 + i < k and col0 + j < n:
                            b_tile[i][j] = b[k0 + i][col0 + j]
                            tile_loads += 1
                for i in range(tile):
                    for j in range(tile):
                        if row0 + i < m and col0 + j < n:
                            for inner in range(tile):
                                c[row0 + i][col0 + j] += a_tile[i][inner] * b_tile[inner][j]
    return c, tile_loads


def main() -> None:
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    reference = naive_gemm(a, b)
    tiled, loads = tiled_gemm(a, b, tile=2)
    print("A[2,3] @ B[3,2] =", reference)
    print("tiled equals reference:", tiled == reference)
    print("valid tile loads (illustrative):", loads)
    print("FLOPs =", 2 * 2 * 2 * 3)


if __name__ == "__main__":
    main()
