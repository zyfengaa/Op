"""把 GPU 线程映射拆成可手算的整数例子。Python 3.10+，无第三方依赖。"""


def ceil_div(n: int, block: int) -> int:
    if n < 0 or block <= 0:
        raise ValueError("n must be >= 0 and block must be > 0")
    return (n + block - 1) // block


def one_dimensional_grid(n: int, block: int):
    for block_id in range(ceil_div(n, block)):
        for thread_id in range(block):
            global_id = block_id * block + thread_id
            yield block_id, thread_id, global_id, global_id < n


def offset(row: int, col: int, stride: int) -> int:
    return row * stride + col


def main() -> None:
    n, block = 10, 4
    items = list(one_dimensional_grid(n, block))
    print(f"N={n}, block={block}, grid={ceil_div(n, block)}, launched={len(items)}")
    for block_id, thread_id, global_id, valid in items:
        print(f"block={block_id} thread={thread_id} index={global_id} valid={valid}")
    print(f"A[2,3] contiguous 5 columns -> offset={offset(2, 3, 5)}")
    print(f"A[2,3] view with stride 8 -> offset={offset(2, 3, 8)}")


if __name__ == "__main__":
    main()
