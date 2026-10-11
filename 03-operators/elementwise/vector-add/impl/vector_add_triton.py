"""Triton Vector Add 教学版本。

依赖需要在有兼容 GPU 的环境中安装。此文件展示编程模型，不宣称已在本机编译或测量。
"""
import torch
import triton
import triton.language as tl


# 说明：本文件从未在真实 GPU 上编译或运行验证过。它是按 Triton 的编程模型写的教学代码，
# 下面的注释解释"若运行应当如此"，不是实测结论。正确性需在兼容 GPU 上自行验证。
@triton.jit
def _add_kernel(a_ptr, b_ptr, c_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    # 与 CUDA 最大的差异：这里没有 threadIdx/blockIdx，也没有逐线程的 if (i < n)。
    # 一个 program 实例一次处理整块 BLOCK 个元素——思考粒度是 tile，不是线程。
    # program_id(0) 对应 CUDA 的 blockIdx.x，即"第几个 tile"，而不是"第几个线程"。
    # 标成 tl.constexpr 的值在编译期被特化进 kernel；代价是换一个 n/BLOCK 就要重新编译。
    pid = tl.program_id(0)
    # tl.arange(0, BLOCK) 生成 [0,1,...,BLOCK-1] 的向量，加上 pid 的基址得到本 tile 的全局下标。
    # 对应 CUDA 里每个线程各自算出的那一个 i，只是这里一次算出一整条。
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    # mask 取代了 CUDA 里手写的边界判断：把"越界元素不参与"变成 load/store 的参数交给语言处理。
    # 注意 mask 保护的是读写本身；other=0 只是给被屏蔽的 lane 一个占位值，不会写回输出。
    mask = offsets < n_elements
    # 一次 load 取回整个 tile，语义上等同于 CUDA 中每线程一次标量访存，只是表达层次更高。
    a = tl.load(a_ptr + offsets, mask=mask, other=0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0)
    # store 同样带 mask，所以尾部多出的 lane 不会越界写 C——host 侧无需再算 tail。
    tl.store(c_ptr + offsets, a + b, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # 对应 CUDA 版放在参数约定层的检查：形状/设备/dtype 不符时应在这里明确报错，
    # 而不是把不匹配的指针交给 kernel 读出乱数据。
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("a and b must share device and dtype")
    # 当前 kernel 只按线性指针 + 偏移索引，这个模型只在 CUDA 张量上成立。
    if not a.is_cuda:
        raise ValueError("this example expects CUDA tensors")
    # Triton 按指针偏移寻址，必须假设内存连续；.contiguous() 保证 strides 是常规布局，
    # 否则 offsets 的线性映射不成立（非连续视图会算错）。
    a_contig, b_contig = a.contiguous(), b.contiguous()
    out = torch.empty_like(a_contig)
    n = out.numel()
    block = 256
    # [(...)] 是 grid 的写法：括号内给出每个维度上有多少个 program 实例。
    # triton.cdiv 做向上取整，等价于 CUDA 的 (n + block - 1) / block；
    # 尾块不足一个 tile 的部分由 kernel 内的 mask 兜住，host 侧不必再单独处理。
    _add_kernel[(triton.cdiv(n, block),)](a_contig, b_contig, out, n, block)
    return out


def main():
    # 与 CUDA 例一样刻意用 256 的非整数倍长度：让最后一个 tile 真的有被 mask 挡掉的 lane。
    n = 100003  # deliberate tail
    # 用随机输入而非常数：常数输入下读错相邻元素往往得到同样结果，会把索引类错误藏住。
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    actual = vector_add(a, b)
    # 用 torch 自身的 a+b 作为 reference，而不是再写一份循环；对拍对象独立于被测 kernel。
    expected = a + b
    # assert_close 不只看数值，还核对 dtype/设备/shape；默认带浮点容差。
    torch.testing.assert_close(actual, expected)
    print("Triton Vector Add correctness: PASS")


if __name__ == "__main__":
    main()
