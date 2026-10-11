"""Measure two operator stories on CPU; Python and PyTorch native backends."""
# 两个"算子故事"（Vector Add 与 Reduce-Sum）的对照实验：同一组逻辑值，
# 分别在纯 Python 循环和 PyTorch 原生后端上跑，比较耗时与误差。
#
# 口径说明（关键，本仓库反复强调"不要混口径"）：
#   * 只比"同设备、同 dtype、同 N、同计时范围"的数字；
#   * "把 Python 列表转成张量再算"的成本被单独列成一个候选（add/torch_with_conversion），
#     不许把它混进"resident 张量"的数字里，否则对比不公平；
#   * 每个 N 单独画图：大 N 的 Python 柱会把小 N 的差异压平，混在一张图上会误导。
import argparse
import math
from pathlib import Path
import sys

# 允许从任意工作目录以脚本方式运行：把仓库根加入 sys.path 以便 import course_runtime。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from course_runtime import bar_svg, metadata, save_json, stats, synchronize
import time


# 纯 Python 参考实现：逐元素相加。它是"正确性基线"，也是"解释器开销"的计时对象。
def python_add(a, b):
    out = []
    for x, y in zip(a, b):
        out.append(x + y)
    return out


# 顺序累加：浮点加法不满足结合律，累加顺序会影响结果，
# 所以它只是一种"口径"，不是唯一真值（下面会和 tree/fsum/torch 对拍）。
def python_sum(x):
    total = 0.0
    for value in x:
        total += value
    return total


# 树形（成对）归约：相邻两两相加，逐层折半。
# 相比顺序累加，它把误差增长从 O(N) 降到 O(log N)，空输入返回中性值 0.0。
# 注意奇数长度时最后一个元素直接晋级到下一层，不能被丢掉。
def tree_sum(x):
    current = list(x)
    if not current:
        return 0.0
    while len(current) > 1:
        current = [current[i] + current[i + 1] if i + 1 < len(current) else current[i]
                   for i in range(0, len(current), 2)]
    return current[0]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("09-practice/operator-stories/results/local"))
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    # 挡掉非正参数：0 次采样会给出无意义的统计量。
    if args.samples <= 0 or args.iterations <= 0:
        parser.error("positive samples and iterations required")
    torch.set_num_threads(1)
    rows, checks = [], []
    # 选 1 / 17 / 1024 / 65536 四个尺度：分别对应"纯解释器开销""小规模""缓存内""远离缓存"。
    for size in (1, 17, 1024, 65536):
        # 用随下标变化的周期输入而不是常数：常数会让"读错相邻元素"也得到相同结果，掩盖 bug。
        a = [float(i % 13 - 6) / 4 for i in range(size)]
        b = [float(i % 7 - 3) / 2 for i in range(size)]
        ta, tb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
        expected = [x + y for x, y in zip(a, b)]
        # Cross-backend same logical values, double precision, new output contract.
        # 先证等价再计时：Python、torch.add，以及三种求和口径必须给出同一组值。
        # 用 float64 是为了让"顺序累加/树形/fsum/torch.sum"在这些输入上精确一致，
        # 不让正确性检查退化成"碰浮点运气"。
        assert python_add(a, b) == expected
        assert torch.add(ta, tb).tolist() == expected
        assert python_sum(a) == tree_sum(a) == math.fsum(a) == torch.sum(ta).item()
        checks.append({"size": size, "vector_add": "PASS", "reduce_sum": "PASS"})
        cases = {
            # 两个"解释器 vs 原生"的对照，外加一个显式的"转换成本"候选。
            "add/python_loop": lambda: python_add(a, b),
            "add/torch_resident": lambda: torch.add(ta, tb),
            # 先把列表转成张量、算完再转回列表：转换成本单独成项，
            # 这样它不会污染上面 resident 张量的对比。
            "add/torch_with_conversion": lambda: torch.add(
                torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)).tolist(),
            "sum/python_loop": lambda: python_sum(a),
            "sum/python_tree": lambda: tree_sum(a),
            "sum/torch_resident": lambda: torch.sum(ta),
        }
        # 预热 3 次：把一次性初始化成本挪出计时范围。
        for fn in cases.values():
            for _ in range(3):
                fn()
        raw = {key: [] for key in cases}
        for sample in range(args.samples):
            # 奇偶样本反转候选顺序，抵消候选之间的顺序偏差。
            keys = list(cases) if sample % 2 == 0 else list(reversed(cases))
            for key in keys:
                start = time.perf_counter_ns()
                for _ in range(args.iterations):
                    cases[key]()
                # 批平均到单次微秒；计时范围是"宿主墙钟"，只含这次调用本身，
                # 不含建表、也不含张量转换——除非该候选本身就是转换候选。
                raw[key].append((time.perf_counter_ns() - start) / 1000 / args.iterations)
        rows.extend({"size": size, "variant": key, **stats(values)} for key, values in raw.items())

    args.output.mkdir(parents=True, exist_ok=True)
    # profiler 单独采集：只对"resident 张量"的两个故事打点，说明原生调用对应哪些事件。
    # 它带插桩开销，其绝对时间不能与上面的微基准混着比。此处用的是循环最后一轮（N=65536）的张量。
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=True, profile_memory=True) as prof:
        with torch.profiler.record_function("story/vector_add"):
            torch.add(ta, tb)
        with torch.profiler.record_function("story/reduce_sum"):
            torch.sum(ta)
    prof.export_chrome_trace(str(args.output / "native-cpu.trace.json"))
    (args.output / "native-cpu.profiler.txt").write_text(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30), encoding="utf-8")
    report = {"metadata": metadata([Path(__file__), Path(__file__).resolve().parents[2] / "course_runtime.py"]),
              "contract": "finite float64 values; sum returns scalar, add returns newly allocated values",
              # measurement 把口径写死：样本数、每样本调用次数、warmup、计时范围、线程数。
              # 读者对比时先比口径、再比数字。
              "measurement": {"samples": args.samples, "calls_per_sample": args.iterations,
                              "warmup": 3, "scope": "CPU host-wall; Python lists versus resident torch tensors; conversion explicitly separate",
                              "threads": 1, "profiler_input_size": 65536},
              "correctness": checks, "rows": rows,
              "native_profile_events": [event.key for event in prof.key_averages()],
              # 本机没有 GPU 侧 ncu 证据，显式写 null，避免把 CPU 结论说成 GPU 结论。
              "gpu_ncu_evidence": None}
    save_json(args.output / "summary.json", report)
    # Plot each shape independently so a large-N Python bar does not hide small-N timings.
    # 每个 N 单独出一张图：大 N 的 Python 柱高达数十微秒，与小 N 同轴会把小 N 的差异压平。
    for size in (1, 17, 1024, 65536):
        bar_svg(args.output / f"n-{size}.svg",
                [(row["variant"], row["median_us"]) for row in rows if row["size"] == size],
                f"Vector Add / Reduce - N={size}",
                "Actual CPU timings; float64; 1 torch thread; conversion cost is a separate candidate")
    for size in (1, 17, 1024, 65536):
        values = {row["variant"]: row["median_us"] for row in rows if row["size"] == size}
        # 打印中位数（而非最小值）：中位数可比、可复现，最小值只是"最走运的一次"。
        print(f"N={size} add Python={values['add/python_loop']:.3f} us vs native={values['add/torch_resident']:.3f} us; "
              f"sum Python={values['sum/python_loop']:.3f} us vs native={values['sum/torch_resident']:.3f} us")
    print("cross-backend correctness: PASS")
    print("report:", args.output / "summary.json")


if __name__ == "__main__":
    main()
