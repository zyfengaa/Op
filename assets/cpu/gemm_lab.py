"""GEMM 手算实验：naive 与 tiled、非整除尺寸、tile load 次数。"""


def validate(a: list[list[float]], b: list[list[float]]) -> tuple[int, int, int]:
    # 空矩阵或空行会让后面取 a[0]/b[0] 直接 IndexError，提前拦下并给出清楚的报错。
    if not a or not b or not a[0] or not b[0]:
        raise ValueError("non-empty matrices required")
    m, k, n = len(a), len(a[0]), len(b[0])
    # 返回值顺序是 (m, n, k)，与这里的读取顺序 (m, k, n) 不同，调用方容易弄反。
    # 参差不齐的行会让 range(k)/range(n) 越界或悄悄取到错位的元素，必须拒绝。
    if any(len(row) != k for row in a) or len(b) != k or any(len(row) != n for row in b):
        raise ValueError("incompatible or ragged matrices")
    return m, n, k


def naive_gemm(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    m, n, k = validate(a, b)
    # 每个输出元素都重新读整行 A 与整列 B —— 没有复用，这正是 tiled 要解决的问题。
    # 用 sum 而非 fsum：累加顺序固定为 inner 从 0 到 k-1，浮点结果与分块累加
    # 可能逐位不同（加法不满足结合律），这一点由测试用例的整数数据规避了。
    return [[sum(a[row][inner] * b[inner][col] for inner in range(k))
             for col in range(n)] for row in range(m)]


def tiled_gemm(
    a: list[list[float]], b: list[list[float]], tile: int
) -> tuple[list[list[float]], int]:
    m, n, k = validate(a, b)
    # tile=0 会让内层 range 全空，输出全 0 且 tile_loads=0 —— 静默的错，必须显式报错。
    if tile <= 0:
        raise ValueError("tile must be positive")
    # 显式零初始化累加器，对应 GPU 里每个线程的寄存器累加器清零。
    c = [[0.0] * n for _ in range(m)]
    # 统计"有效装载"次数，用来说明 tile 越大重复读越少；它不是真实带宽/耗时。
    tile_loads = 0
    # 三重 tile 循环：row0/col0 遍历输出 tile，k0 把 K 维分块。
    for row0 in range(0, m, tile):
        for col0 in range(0, n, tile):
            for k0 in range(0, k, tile):
                # 每个 K tile 重新申请 buffer，模拟共享内存 tile 被反复覆盖。
                a_tile = [[0.0] * tile for _ in range(tile)]
                b_tile = [[0.0] * tile for _ in range(tile)]
                for i in range(tile):
                    for j in range(tile):
                        # 越界不装载，留 0。0 是乘法的中性元：A 侧补 0 等价于
                        # K 方向补零，不改变结果（与 gemm_first_tile_trace 同理）。
                        if row0 + i < m and k0 + j < k:
                            a_tile[i][j] = a[row0 + i][k0 + j]
                            tile_loads += 1
                        # b_tile 是 K×N 方向：i 对应 K，j 对应 N，边界条件因此不同。
                        if k0 + i < k and col0 + j < n:
                            b_tile[i][j] = b[k0 + i][col0 + j]
                            # 只统计真正装载的元素，padding 的 0 不计，否则计数虚高、
                            # 看不出复用效果。
                            tile_loads += 1
                for i in range(tile):
                    for j in range(tile):
                        # 输出越界的线程不写回，避免覆盖别的 tile 已经算好的部分和
                        # （对应 GM01 的越界线程退出与 VA01 的尾部保护）。
                        if row0 + i < m and col0 + j < n:
                            for inner in range(tile):
                                c[row0 + i][col0 + j] += a_tile[i][inner] * b_tile[inner][j]
    return c, tile_loads


def main() -> None:
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    reference = naive_gemm(a, b)
    # M=2, N=2, K=3，tile=2：K 方向有尾巴，用来说明 padding 不影响结果。
    tiled, loads = tiled_gemm(a, b, tile=2)
    # 用 == 精确比较是安全的：这里全是整数输入，两种实现的加法顺序不引入误差。
    print("A[2,3] @ B[3,2] =", reference)
    print("tiled equals reference:", tiled == reference)
    # 注意这是"有效装载次数"，不是真实访存成本，只作示意。
    print("valid tile loads (illustrative):", loads)
    # FLOPs = 2（每对乘加）* M * N * K。
    print("FLOPs =", 2 * 2 * 2 * 3)


if __name__ == "__main__":
    main()
