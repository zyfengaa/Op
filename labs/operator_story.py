"""Measure two operator stories on CPU; Python and PyTorch native backends."""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from course_runtime import bar_svg, metadata, save_json, stats, synchronize
import time


def python_add(a, b):
    out = []
    for x, y in zip(a, b):
        out.append(x + y)
    return out


def python_sum(x):
    total = 0.0
    for value in x:
        total += value
    return total


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
    parser.add_argument("--output", type=Path, default=Path("labs/results/local"))
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.samples <= 0 or args.iterations <= 0:
        parser.error("positive samples and iterations required")
    torch.set_num_threads(1)
    rows, checks = [], []
    for size in (1, 17, 1024, 65536):
        a = [float(i % 13 - 6) / 4 for i in range(size)]
        b = [float(i % 7 - 3) / 2 for i in range(size)]
        ta, tb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
        expected = [x + y for x, y in zip(a, b)]
        # Cross-backend same logical values, double precision, new output contract.
        assert python_add(a, b) == expected
        assert torch.add(ta, tb).tolist() == expected
        assert python_sum(a) == tree_sum(a) == math.fsum(a) == torch.sum(ta).item()
        checks.append({"size": size, "vector_add": "PASS", "reduce_sum": "PASS"})
        cases = {
            "add/python_loop": lambda: python_add(a, b),
            "add/torch_resident": lambda: torch.add(ta, tb),
            "add/torch_with_conversion": lambda: torch.add(
                torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)).tolist(),
            "sum/python_loop": lambda: python_sum(a),
            "sum/python_tree": lambda: tree_sum(a),
            "sum/torch_resident": lambda: torch.sum(ta),
        }
        for fn in cases.values():
            for _ in range(3):
                fn()
        raw = {key: [] for key in cases}
        for sample in range(args.samples):
            keys = list(cases) if sample % 2 == 0 else list(reversed(cases))
            for key in keys:
                start = time.perf_counter_ns()
                for _ in range(args.iterations):
                    cases[key]()
                raw[key].append((time.perf_counter_ns() - start) / 1000 / args.iterations)
        rows.extend({"size": size, "variant": key, **stats(values)} for key, values in raw.items())

    args.output.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=True, profile_memory=True) as prof:
        with torch.profiler.record_function("story/vector_add"):
            torch.add(ta, tb)
        with torch.profiler.record_function("story/reduce_sum"):
            torch.sum(ta)
    prof.export_chrome_trace(str(args.output / "native-cpu.trace.json"))
    (args.output / "native-cpu.profiler.txt").write_text(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30), encoding="utf-8")
    report = {"metadata": metadata([Path(__file__), Path(__file__).resolve().parents[1] / "course_runtime.py"]),
              "contract": "finite float64 values; sum returns scalar, add returns newly allocated values",
              "measurement": {"samples": args.samples, "calls_per_sample": args.iterations,
                              "warmup": 3, "scope": "CPU host-wall; Python lists versus resident torch tensors; conversion explicitly separate",
                              "threads": 1, "profiler_input_size": 65536},
              "correctness": checks, "rows": rows,
              "native_profile_events": [event.key for event in prof.key_averages()],
              "gpu_ncu_evidence": None}
    save_json(args.output / "summary.json", report)
    # Plot each shape independently so a large-N Python bar does not hide small-N timings.
    for size in (1, 17, 1024, 65536):
        bar_svg(args.output / f"n-{size}.svg",
                [(row["variant"], row["median_us"]) for row in rows if row["size"] == size],
                f"Vector Add / Reduce - N={size}",
                "Actual CPU timings; float64; 1 torch thread; conversion cost is a separate candidate")
    for size in (1, 17, 1024, 65536):
        values = {row["variant"]: row["median_us"] for row in rows if row["size"] == size}
        print(f"N={size} add Python={values['add/python_loop']:.3f} us vs native={values['add/torch_resident']:.3f} us; "
              f"sum Python={values['sum/python_loop']:.3f} us vs native={values['sum/torch_resident']:.3f} us")
    print("cross-backend correctness: PASS")
    print("report:", args.output / "summary.json")


if __name__ == "__main__":
    main()
