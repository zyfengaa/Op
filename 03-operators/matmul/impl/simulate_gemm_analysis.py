"""演示 Decode 小 M GEMM 的性能分析计算。

本脚本只使用 Python 标准库，所有输入都是教学用模拟数据，不是硬件实测。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GemmCase:
    m: int
    n: int
    k: int
    bytes_per_element: int
    peak_tflops: float
    peak_bandwidth_tbps: float

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k

    @property
    def weight_bytes(self) -> int:
        return self.k * self.n * self.bytes_per_element

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.weight_bytes

    @property
    def roofline_knee(self) -> float:
        return (self.peak_tflops * 1_000) / self.peak_bandwidth_tbps


def mib(value: int) -> float:
    return value / (1024 * 1024)


def main() -> None:
    case = GemmCase(
        m=1,
        n=4096,
        k=4096,
        bytes_per_element=2,
        peak_tflops=100,
        peak_bandwidth_tbps=1.5,
    )

    print("=== Simulated Decode GEMM analysis ===")
    print(f"shape: [{case.m}, {case.k}] x [{case.k}, {case.n}]")
    print(f"FLOPs: {case.flops / 1_000_000:.2f} MFLOPs")
    print(f"weight traffic: {mib(case.weight_bytes):.2f} MiB")
    print(f"arithmetic intensity: {case.arithmetic_intensity:.2f} FLOP/Byte")
    print(f"Roofline knee: {case.roofline_knee:.2f} FLOP/Byte")
    bound = "Memory Bound" if case.arithmetic_intensity < case.roofline_knee else "Compute Bound"
    print(f"initial classification: {bound}")

    before_us = 82.0
    after_us = 43.0
    print(f"kernel speedup: {before_us / after_us:.2f}x ({before_us:.0f} -> {after_us:.0f} us)")

    e2e_before_us = 320.0
    e2e_gemm_only_us = 281.0
    e2e_fused_us = 263.0
    print(f"E2E GEMM-only improvement: {(1 - e2e_gemm_only_us / e2e_before_us) * 100:.1f}%")
    print(f"E2E with fusion improvement: {(1 - e2e_fused_us / e2e_before_us) * 100:.1f}%")


if __name__ == "__main__":
    main()
