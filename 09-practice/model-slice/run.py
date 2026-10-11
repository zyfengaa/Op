"""Run, compare and profile the full prefill/decode block. See README.md."""
# 端到端脚本：跑通完整 block，做正确性对照、计时、可选 profiler 与内存记录，
# 并把结果写成 summary.json 与 SVG 图。README.md 解释每一项分别在回答什么问题。
#
# 贯穿全脚本的口径纪律（本仓库反复强调的一点）：
#   * 只比较"同设备、同 dtype、同 shape、同计时范围"下的数字；
#   * 报中位数和区间（min..max），不挑单次最小值——最小值是运气，中位数才可复现；
#   * 计时范围包含分配与 cache 拼接，不含模型构造、分词、H2D/D2H、词表 head；
#   * profiler 是另跑一次、带插桩的调用，其绝对数字不能与微基准混着比。
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

# 允许从任意工作目录以脚本方式运行：把仓库根加入 sys.path，
# 这样才能 import 到 course_runtime 和同目录的 block。
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
    # 提前挡掉非法参数，避免出现"空张量"或"0 次采样"这类只会给出无意义数字的配置。
    if min(args.seq, args.batch, args.threads, args.samples, args.iterations) <= 0:
        parser.error("sequence, batch, threads, samples and iterations must be positive")
    # 明确拒绝"声称用 CUDA 但环境没有 CUDA"：否则会静默退回 CPU，数字张冠李戴。
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("this PyTorch environment has no available CUDA device")
    torch.set_num_threads(args.threads)
    cfg = Config(args.width, args.heads, args.hidden)
    dtype = getattr(torch, args.dtype)
    model = TinyDecoderBlock(cfg, device=args.device, dtype=dtype)
    # 固定 seed 生成输入：不同次运行、不同 variant 都用完全相同的输入，比较才有效。
    generator = torch.Generator().manual_seed(18)
    full_x = torch.randn(args.batch, args.seq + 1, cfg.width, generator=generator, dtype=dtype).to(args.device)
    # 拆成"长度为 seq 的前缀 + 1 个 token"，同时驱动 prefill 与单 token decode。
    prefix, token = full_x[:, :-1], full_x[:, -1:]
    # 期望值由 reference 整段前向给出，后面所有 variant 都对齐它。
    expected, _ = model(full_x, variant="reference")
    # base_cache 是"不可变的前缀缓存"：多次计时迭代都复用它，防止 cache 越测越长。
    _, base_cache = model(prefix, variant="reference")
    cases = {}
    # 容差随精度走：float32 累加误差大，放宽到 2e-5；float64 收严到 1e-10。
    atol = rtol = 2e-5 if dtype == torch.float32 else 1e-10
    correctness = []
    for variant in ("reference", "optimized"):
        # 先证等价再计时：prefill 段与 decode 段分别对齐 reference 的整段结果。
        pre, _ = model(prefix, variant=variant)
        dec, extended = model(token, base_cache, variant=variant)
        torch.testing.assert_close(pre, expected[:, :-1], atol=atol, rtol=rtol)
        torch.testing.assert_close(dec, expected[:, -1:], atol=atol, rtol=rtol)
        # 顺手校验缓存长度契约：新缓存应比前缀多 1，且基准缓存本身保持不变。
        assert extended.length == args.seq + 1 and base_cache.length == args.seq
        correctness.append({"variant": variant, "prefill_and_decode": "PASS",
                            "decode_max_abs_error": float((dec - expected[:, -1:]).abs().max())})
        # 计时对象是"整块 block 的一次调用"，不是单个算子；decode 复用不可变的前缀 cache。
        cases[f"prefill/{variant}"] = lambda profile=False, variant=variant: model(
            prefix, variant=variant, profile=profile)
        cases[f"decode/{variant}"] = lambda profile=False, variant=variant: model(
            token, base_cache, variant=variant, profile=profile)

    # Warm up every candidate, then interleave measurements to reduce ordering bias.
    # 预热 3 次：让 lazy 初始化、首次分配、缓存预热等一次性成本落在计时范围之外。
    for fn in cases.values():
        for _ in range(3):
            fn()
    synchronize(args.device)
    raw = {key: [] for key in cases}
    for sample in range(args.samples):
        # 奇偶样本反转候选顺序：抵消"后测的候选受前一候选影响"这类顺序偏差。
        keys = list(cases) if sample % 2 == 0 else list(reversed(cases))
        for key in keys:
            # 起始同步：确保测到的是计算本身，而不是上一任务尚未跑完的排队时间。
            synchronize(args.device)
            start = time.perf_counter_ns()
            for _ in range(args.iterations):
                cases[key]()
            # 结束同步：CUDA 是异步发射，不同步计到的只是"入队"耗时，不是真正跑完的时间。
            synchronize(args.device)
            # 除以 iterations 得到单次调用的微秒数；每个 sample 是一个"批平均"。
            raw[key].append((time.perf_counter_ns() - start) / 1000 / args.iterations)

    args.output.mkdir(parents=True, exist_ok=True)
    memory, profiles = {}, {}
    for key, fn in cases.items():
        if args.device.startswith("cuda"):
            # 记录 allocator 口径的显存：baseline 是调用前的已分配量，peak 是本次调用峰值，
            # 两者之差才是"这次 block 调用额外吃掉的显存"。这是 PyTorch allocator 视角，
            # 不等于 nvidia-smi 的进程占用（缓存与碎片要另算）。
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
            # 及时释放，避免上一候选的显存占用抬高下一候选的 baseline。
            del output, cache
        else:
            # CPU 路径没有 CUDA 显存可测，字段留 null，只给可手算的"逻辑 KV 字节数"。
            # 明确写出"未测 CPU 原生峰值"，以免读者误以为峰值就只有这么点。
            length = args.seq if key.startswith("prefill") else args.seq + 1
            memory[key] = {"cuda_peak_allocated_bytes": None, "cpu_peak_rss_bytes": None,
                           "status": "CPU run: no CUDA memory; CPU native peak not measured",
                           "logical_output_kv_bytes": 2 * args.batch * cfg.width * length * full_x.element_size()}
        if args.profile:
            # profiler 是"另跑一次"的采集：它给每个 slice/ 阶段插桩，自带额外开销，
            # 所以它的绝对时间不能和上面的微基准混着比，只能看阶段内的相对占比。
            activities = [torch.profiler.ProfilerActivity.CPU]
            if args.device.startswith("cuda"):
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                fn(profile=True)
                synchronize(args.device)
            stem = key.replace("/", "-")
            # 保留原始 trace 与文本表，便于复核"这个结论到底出自哪个事件"。
            prof.export_chrome_trace(str(args.output / f"{stem}.trace.json"))
            averages = prof.key_averages()
            (args.output / f"{stem}.profiler.txt").write_text(
                averages.table(sort_by="self_cpu_time_total", row_limit=40), encoding="utf-8")
            stages = [{"stage": event.key.removeprefix("slice/"), "calls": event.count,
                       "cpu_total_us": event.cpu_time_total,
                       "device_total_us": getattr(event, "device_time_total", 0.0)}
                      for event in averages if event.key.startswith("slice/")]
            # 占比分母是本次采集内这些阶段 CPU 时间之和，是"阶段间相对占比"，
            # 不是 GPU SM 利用率；也不要把嵌套事件的 total 直接相加。
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
        # metadata 记下参与文件的内容哈希与设备，让这份报告能追溯到具体代码版本。
        "metadata": metadata([Path(__file__), Path(__file__).with_name("block.py"),
                              Path(__file__).resolve().parents[2] / "course_runtime.py"], args.device),
        "config": asdict(cfg), "batch": args.batch, "prefix_length": args.seq, "dtype": args.dtype,
        # timing 里放的是统计量（中位数/区间），不是单次原始值；原始样本已在 raw 里消费掉。
        "correctness": correctness, "timing": {key: stats(value) for key, value in raw.items()},
        # measurement 把口径写死在报告里：warmup 次数、样本数、每样本调用几次、
        # 计时范围含什么不含什么。读者拿去对比时，先比口径，再比数字。
        "measurement": {"warmup": 3, "samples": args.samples, "calls_per_sample": args.iterations,
                        "scope": "host-wall full block on resident inputs; CUDA synchronized when used; allocations and cache concatenation included",
                        "decode": "one token against a fixed immutable prefix cache; cache does not grow between timing iterations",
                        "not_included": "model construction, checkpoint/tokenization, H2D/D2H, language-model head"},
        "shapes": {"prefill": shapes(cfg, args.batch, args.seq),
                   "decode": shapes(cfg, args.batch, 1, args.seq)},
        "memory": memory, "profiles": profiles,
        # SDPA 由 PyTorch 按环境分派；要判断实际后端得看 native_events，不能假定是 FlashAttention。
        "attention_backend": "PyTorch SDPA automatic dispatch; inspect native_events, do not assume GPU FlashAttention",
    }
    save_json(args.output / "summary.json", report)
    # E2E 图用中位数而不是最小值：最小值只是最顺的一次，不能代表典型耗时。
    bar_svg(args.output / "e2e.svg", [(key, statistics["median_us"]) for key, statistics in report["timing"].items()],
            "Tiny decoder block - measured E2E", f"{report['metadata']['device']}, {args.dtype}, median of batch-averages")
    # 打印"中位数 + 区间"：区间让人看到抖动幅度，避免把一次尖峰或一次幸运值当成结论。
    for key, values in report["timing"].items():
        print(f"{key}: median={values['median_us']:.3f} us, range={values['min_us']:.3f}..{values['max_us']:.3f}")
    print("prefill/decode correctness: PASS")
    print("report:", args.output / "summary.json")


if __name__ == "__main__":
    main()
