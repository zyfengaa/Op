"""把 GPU 线程映射拆成可手算的整数例子。Python 3.10+，无第三方依赖。"""


def ceil_div(n: int, block: int) -> int:
    # n<0 会得到 0 或负的 grid；block<=0 会直接除零。两者都要显式挡住。
    if n < 0 or block <= 0:
        raise ValueError("n must be >= 0 and block must be > 0")
    # 整数除法向下取整，先把被除数补到 block 的整数倍，最后一截才不会被漏掉。
    # 写成 n // block 会少发最后一个 block —— 这正是故障题 VA01 的症状来源。
    # n==0 时返回 0：没有元素就不发射任何 block。
    return (n + block - 1) // block


def one_dimensional_grid(n: int, block: int):
    # 用生成器模拟 <<<grid, block>>> 的发射：外层 block_id、内层 thread_id，
    # 与 GPU 中 (blockIdx.x, threadIdx.x) 的二维展开顺序一致。
    for block_id in range(ceil_div(n, block)):
        for thread_id in range(block):
            # 与 CUDA 的 blockIdx.x * blockDim.x + threadIdx.x 完全同形。
            global_id = block_id * block + thread_id
            # valid 就是 kernel 里那个 if (i < n)。关键点：越界线程在这里依然被
            # yield 出来，只标 valid=False —— 真实 GPU 上这些线程确实被发射了，
            # 只是不写回。把 valid 当成"线程不存在"就会漏掉尾部的保护逻辑。
            yield block_id, thread_id, global_id, global_id < n


def offset(row: int, col: int, stride: int) -> int:
    # 一维地址 = 行号 × 行跨度 + 列号。stride 是"相邻两行起点相隔多少个元素"：
    # 连续张量里 stride 等于列数，切片视图里可以更大（跳着读）——对应故障题 VA02。
    return row * stride + col


def main() -> None:
    # N=10 不是 block=4 的整数倍：12 个线程被发射，其中 2 个越界。
    n, block = 10, 4
    items = list(one_dimensional_grid(n, block))
    print(f"N={n}, block={block}, grid={ceil_div(n, block)}, launched={len(items)}")
    for block_id, thread_id, global_id, valid in items:
        print(f"block={block_id} thread={thread_id} index={global_id} valid={valid}")
    # 同一个 A[2,3]，stride 不同（5 vs 8）会得到不同的线性地址：
    # 前者是连续张量，后者是带行填充/切片视角的视图。
    print(f"A[2,3] contiguous 5 columns -> offset={offset(2, 3, 5)}")
    print(f"A[2,3] view with stride 8 -> offset={offset(2, 3, 8)}")


if __name__ == "__main__":
    main()
