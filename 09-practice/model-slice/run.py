"""Run, compare and profile the full prefill/decode block. See README.md."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from block import Config, TinyDecoderBlock, shapes
from course_runtime import bar_svg, metadata, save_json, stats, synchronize


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("09-practice/model-slice/results/local"))
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.seq, args.batch, args.threads, args.samples, args.iterations) <= 0:
        parser.error("sequence, batch, threads, samples and iterations must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("this PyTorch environment has no available CUDA device")
    torch.set_num_threads(args.threads)
    cfg = Config(args.width, args.heads, args.hidden)
    dtype = getattr(torch, args.dtype)
    model = TinyDecoderBlock(cfg, device=args.device, dtype=dtype)
    generator = torch.Generator().manual_seed(18)
    full_x = torch.randn(args.batch, args.seq + 1, cfg.width, generator=generator, dtype=dtype).to(args.device)
    prefix, token = full_x[:, :-1], full_x[:, -1:]
    expected, _ = model(full_x, variant="reference")
    _, base_cache = model(prefix, variant="reference")
    cases = {}
    atol = rtol = 2e-5 if dtype == torch.float32 else 1e-10
    correctness = []
    for variant in ("reference", "optimized"):
        pre, _ = model(prefix, variant=variant)
        dec, extended = model(token, base_cache, variant=variant)
        torch.testing.assert_close(pre, expected[:, :-1], atol=atol, rtol=rtol)
        torch.testing.assert_close(dec, expected[:, -1:], atol=atol, rtol=rtol)
        assert extended.length == args.seq + 1 and base_cache.length == args.seq
        correctness.append({"variant": variant, "prefill_and_decode": "PASS",
                            "decode_max_abs_error": float((dec - expected[:, -1:]).abs().max())})
        cases[f"prefill/{variant}"] = lambda profile=False, variant=variant: model(
            prefix, variant=variant, profile=profile)
        cases[f"decode/{variant}"] = lambda profile=False, variant=variant: model(
            token, base_cache, variant=variant, profile=profile)

    # Warm up every candidate, then interleave measurements to reduce ordering bias.
    for fn in cases.values():
        for _ in range(3):
            fn()
    synchronize(args.device)
    raw = {key: [] for key in cases}
    for sample in range(args.samples):
        keys = list(cases) if sample % 2 == 0 else list(reversed(cases))
        for key in keys:
            synchronize(args.device)
            start = time.perf_counter_ns()
            for _ in range(args.iterations):
                cases[key]()
            synchronize(args.device)
            raw[key].append((time.perf_counter_ns() - start) / 1000 / args.iterations)

    args.output.mkdir(parents=True, exist_ok=True)
    memory, profiles = {}, {}
    for key, fn in cases.items():
        if args.device.startswith("cuda"):
            synchronize(args.device)
            baseline = torch.cuda.memory_allocated(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            output, cache = fn()
            synchronize(args.device)
            peak = torch.cuda.max_memory_allocated(args.device)
            memory[key] = {"cuda_baseline_allocated_bytes": baseline,
                           "cuda_peak_allocated_bytes": peak,
                           "cuda_peak_increase_bytes": peak - baseline,
                           "logical_output_kv_bytes": (cache.k.numel() + cache.v.numel()) * cache.k.element_size()}
            del output, cache
        else:
            length = args.seq if key.startswith("prefill") else args.seq + 1
            memory[key] = {"cuda_peak_allocated_bytes": None, "cpu_peak_rss_bytes": None,
                           "status": "CPU run: no CUDA memory; CPU native peak not measured",
                           "logical_output_kv_bytes": 2 * args.batch * cfg.width * length * full_x.element_size()}
        if args.profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if args.device.startswith("cuda"):
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                fn(profile=True)
                synchronize(args.device)
            stem = key.replace("/", "-")
            prof.export_chrome_trace(str(args.output / f"{stem}.trace.json"))
            averages = prof.key_averages()
            (args.output / f"{stem}.profiler.txt").write_text(
                averages.table(sort_by="self_cpu_time_total", row_limit=40), encoding="utf-8")
            stages = [{"stage": event.key.removeprefix("slice/"), "calls": event.count,
                       "cpu_total_us": event.cpu_time_total,
                       "device_total_us": getattr(event, "device_time_total", 0.0)}
                      for event in averages if event.key.startswith("slice/")]
            denominator = sum(row["cpu_total_us"] for row in stages)
            for row in stages:
                row["share_of_instrumented_stage_cpu_time_pct"] = 100 * row["cpu_total_us"] / denominator
            profiles[key] = {"stage_ranges": stages,
                             "native_events": [event.key for event in averages if event.key.startswith("aten::")],
                             "share_scope": "sum of nonoverlapping CPU stage ranges inside this profiled call; not unprofiled E2E"}
            bar_svg(args.output / f"{stem}.stages.svg",
                    [(row["stage"], row["cpu_total_us"]) for row in stages],
                    f"{key} - measured CPU stage ranges",
                    "Profiler-instrumented call; includes instrumentation overhead; not GPU utilization")

    report = {
        "metadata": metadata([Path(__file__), Path(__file__).with_name("block.py"),
                              Path(__file__).resolve().parents[2] / "course_runtime.py"], args.device),
        "config": asdict(cfg), "batch": args.batch, "prefix_length": args.seq, "dtype": args.dtype,
        "correctness": correctness, "timing": {key: stats(value) for key, value in raw.items()},
        "measurement": {"warmup": 3, "samples": args.samples, "calls_per_sample": args.iterations,
                        "scope": "host-wall full block on resident inputs; CUDA synchronized when used; allocations and cache concatenation included",
                        "decode": "one token against a fixed immutable prefix cache; cache does not grow between timing iterations",
                        "not_included": "model construction, checkpoint/tokenization, H2D/D2H, language-model head"},
        "shapes": {"prefill": shapes(cfg, args.batch, args.seq),
                   "decode": shapes(cfg, args.batch, 1, args.seq)},
        "memory": memory, "profiles": profiles,
        "attention_backend": "PyTorch SDPA automatic dispatch; inspect native_events, do not assume GPU FlashAttention",
    }
    save_json(args.output / "summary.json", report)
    bar_svg(args.output / "e2e.svg", [(key, statistics["median_us"]) for key, statistics in report["timing"].items()],
            "Tiny decoder block - measured E2E", f"{report['metadata']['device']}, {args.dtype}, median of batch-averages")
    for key, values in report["timing"].items():
        print(f"{key}: median={values['median_us']:.3f} us, range={values['min_us']:.3f}..{values['max_us']:.3f}")
    print("prefill/decode correctness: PASS")
    print("report:", args.output / "summary.json")


if __name__ == "__main__":
    main()
