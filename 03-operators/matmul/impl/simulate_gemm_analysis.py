"""演示 Decode 小 M GEMM 的性能分析计算。

本脚本只使用 Python 标准库，所有输入都是教学用模拟数据，不是硬件实测。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GemmCase:
    # frozen 让实例不可变：分析用的 shape/峰值参数一旦定下就不该被中途改掉，
    # 否则算出的 AI 和 roofline 会和实际使用的参数对不上。
    m: int
    n: int
    k: int
    # 单个元素的字节数（FP16 = 2）决定权重流量。显式列为参数，是为了让
    # "精度 ↔ 流量"的权衡可直接参与计算，而不是藏进常量里。
    bytes_per_element: int
    # 峰值算力/带宽来自规格书或既有测量报告的假定值，是 roofline 的分母，
    # 不是本脚本测出来的。
    peak_tflops: float
    peak_bandwidth_tbps: float

    @property
    def flops(self) -> int:
        # 2*M*N*K：每个输出元素 K 次乘加，每次乘加按 2 FLOP 计。这是理论计算量下界。
        return 2 * self.m * self.n * self.k

    @property
    def weight_bytes(self) -> int:
        # decode 场景 M=1，把权重算作"理想上只读一次"的最少流量（K*N 个元素）。
        # 这是流量下界，实际是否达到要看 kernel 的复用是否做好。
        return self.k * self.n * self.bytes_per_element

    @property
    def arithmetic_intensity(self) -> float:
        # 算术强度 = FLOP / 搬运字节数，即 roofline 图的横轴。
        return self.flops / self.weight_bytes

    @property
    def roofline_knee(self) -> float:
        # 屋顶线拐点 = 峰值算力 / 峰值带宽，横轴刻度为 FLOP/Byte。
        # 这里的 1_000 是一个量纲换算系数；它只用于判断本 case 落在拐点左侧还是右侧，
        # 属于教学用的简化刻度，不要当作可直接引用的精确拐点。
        return (self.peak_tflops * 1_000) / self.peak_bandwidth_tbps


def mib(value: int) -> float:
    # 统一用 MiB（1024 进制）而非 MB：和内存/显存容量口径一致，
    # 避免 1e6 与 2^20 混用导致不同报告里的数字对不上。
    return value / (1024 * 1024)


def main() -> None:
    case = GemmCase(
        # M=1 是 decode 阶段逐 token 生成的最不利复用情形，权重几乎不被复用。
        m=1,
        n=4096,
        k=4096,
        bytes_per_element=2,  # FP16
        peak_tflops=100,  # 假定值，来自规格书，不是本机测量
        peak_bandwidth_tbps=1.5,  # 同上
    )

    print("=== Simulated Decode GEMM analysis ===")
    print(f"shape: [{case.m}, {case.k}] x [{case.k}, {case.n}]")
    print(f"FLOPs: {case.flops / 1_000_000:.2f} MFLOPs")
    print(f"weight traffic: {mib(case.weight_bytes):.2f} MiB")
    print(f"arithmetic intensity: {case.arithmetic_intensity:.2f} FLOP/Byte")
    print(f"Roofline knee: {case.roofline_knee:.2f} FLOP/Byte")
    bound = "Memory Bound" if case.arithmetic_intensity < case.roofline_knee else "Compute Bound"
    # 这个分类是"根据假定的峰值参数推出来的假设"，用来指导下一步怎么优化；
    # 要下结论必须用 profiler 实测 traffic 和耗时，而不是采信这里的初判。
    print(f"initial classification: {bound}")

    # 以下 before/after 都是写死的模拟数字，只为演示"加速比怎么算"。
    # 它们不代表本仓库任何一次真实测量，不能当作性能结论引用。
    before_us = 82.0
    after_us = 43.0
    print(f"kernel speedup: {before_us / after_us:.2f}x ({before_us:.0f} -> {after_us:.0f} us)")

    # 区分 kernel-only 与 E2E 两种口径：融合能进一步压低端到端时间，
    # 但绝不能把两个口径的改善混成一个"总加速比"来宣称。
    e2e_before_us = 320.0
    e2e_gemm_only_us = 281.0
    e2e_fused_us = 263.0
    print(f"E2E GEMM-only improvement: {(1 - e2e_gemm_only_us / e2e_before_us) * 100:.1f}%")
    print(f"E2E with fusion improvement: {(1 - e2e_fused_us / e2e_before_us) * 100:.1f}%")


if __name__ == "__main__":
    main()
