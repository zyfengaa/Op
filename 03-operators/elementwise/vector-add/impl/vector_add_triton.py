"""Triton Vector Add 教学版本。

依赖需要在有兼容 GPU 的环境中安装。此文件展示编程模型，不宣称已在本机编译或测量。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(a_ptr, b_ptr, c_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("a and b must share device and dtype")
    if not a.is_cuda:
        raise ValueError("this example expects CUDA tensors")
    a_contig, b_contig = a.contiguous(), b.contiguous()
    out = torch.empty_like(a_contig)
    n = out.numel()
    block = 256
    _add_kernel[(triton.cdiv(n, block),)](a_contig, b_contig, out, n, block)
    return out


def main():
    n = 100003  # deliberate tail
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    actual = vector_add(a, b)
    expected = a + b
    torch.testing.assert_close(actual, expected)
    print("Triton Vector Add correctness: PASS")


if __name__ == "__main__":
    main()
