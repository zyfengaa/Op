"""Measurement utilities for executable course reports (not imported by fault labs)."""
import hashlib
import html
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch


def metadata(paths, device="cpu"):
    # 目的：让一份测量报告能回答"这些数字是哪份代码、在什么环境下跑出来的"。
    # 只存数字而没有环境指纹的报告无法追溯，所以这里把两者一起固化成元信息。
    root = Path(__file__).resolve().parent
    def git(*args):
        # git 取不到时返回占位串而不是抛异常：采集环境信息本身不该让测量流程失败。
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    return {
        # 用 UTC 时间戳，避免不同时区/夏令时下两份报告的时间不可比。
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # 记录运行时版本：同一份脚本在别的 python/torch 下性能不可直接比较。
        "python": platform.python_version(), "torch": torch.__version__,
        "os": platform.platform(), "processor": platform.processor(),
        # 非 CUDA 设备明确记成 "CPU"，而不是留空——留空会被误读成漏采。
        "device": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else "CPU",
        # torch_threads 影响 CPU 侧并行度；git_commit 锁定源码版本。
        "torch_threads": torch.get_num_threads(), "git_commit": git("rev-parse", "HEAD"),
        # git_dirty 为真表示工作区有未提交改动，此时 commit hash 不足以代表真正跑的代码，
        # 报告的"可复现"程度要相应打折。
        "git_dirty": bool(git("status", "--porcelain")),
        # 逐个源文件算 sha256：即使忘了提交，事后也能比对出报告对应的是哪一版代码。
        # 路径相对 root 并统一成 "/"，保证 Windows/Linux 产出的 key 一致、可跨机比对。
        "source_sha256": {str(Path(p).relative_to(root)).replace("\\", "/"):
                           hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        # torch 的编译配置决定了它启用了哪些后端/kernel，是性能差异的常见来源之一。
        "torch_build": torch.__config__.show(),
    }


def synchronize(device):
    # 只有 CUDA 的异步执行才需要在计时点前后对齐；CPU 上计时本就同步。
    # 分支让上层脚本对两种设备走同一套调用，又不会在 CPU 上误调 cuda 同步。
    # str() 包一层是为了同时接受 "cuda:0" 字符串和 torch.device 对象。
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def samples(fn, device="cpu", count=9, iterations=10, warmup=3):
    # warmup 不进结果：首次迭代要付 kernel 编译、缓存预热和 GPU 升频的代价，
    # 算进平均会系统性拉高耗时，让人把启动开销误当成稳态性能。
    for _ in range(warmup):
        fn()
    synchronize(device)
    values = []
    for _ in range(count):
        # 采样前同步，隔离上一轮残留的异步工作，避免它串进本次计时窗口。
        synchronize(device)
        # perf_counter_ns 单调且分辨率高，不受系统时间调整影响，适合测微秒级 kernel。
        start = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        # 停表前再同步一次：否则 CPU 可能在 GPU 还没干完时就结束计时，把耗时报小。
        synchronize(device)
        # ns→us 除以 1000，再除以 iterations 得到单次耗时；多轮取值以摊掉逐次抖动。
        values.append((time.perf_counter_ns() - start) / 1000 / iterations)
    return values


def stats(values):
    # median 抗离群点，min 更接近设备满速时的情形，max 暴露最坏抖动。
    # 三者并列给出而不是只报均值：中位数与极值差距越小，这组测量才越可信。
    return {"samples_us": values, "median_us": statistics.median(values),
            "min_us": min(values), "max_us": max(values)}


def save_json(path, data):
    # 主动创建父目录，调用方就不必先手动 mkdir。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False 让中文标题按原文写入，报告可直接阅读；结尾补换行符合文本文件惯例。
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def bar_svg(path, rows, title, subtitle):
    """Simple exported measurement plot. Values must be measured, not hypotheses."""
    # 导出静态 SVG 而不是依赖绘图库：报告在没装 matplotlib 的机器上也能直接打开。
    width, left, right, top, line = 1000, 360, 110, 95, 34
    height = top + len(rows) * line + 45
    # 空 rows 或全 0 会让最大值为 0，导致下面归一化时除零；default 与 or 1 双重兜底。
    maximum = max((value for _, value in rows), default=1) or 1
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f7f9fc"/>',
            '<g font-family="Arial, sans-serif" fill="#172b4d">',
            f'<text x="24" y="32" font-size="22">{html.escape(title)}</text>',
            f'<text x="24" y="59" font-size="13">{html.escape(subtitle)}</text>']
    for index, (label, value) in enumerate(rows):
        y = top + index * line
        # 条长按最大值归一化到绘图区宽度：图上读到的只是相对高低，绝对量值由右侧文字给出。
        bar = value / maximum * (width - left - right)
        # 标题与标签都经 html.escape 转义，避免其中的 < > & 破坏 SVG 结构。
        body += [f'<text x="24" y="{y + 16}" font-size="13">{html.escape(label)}</text>',
                 f'<rect x="{left}" y="{y}" width="{bar:.3f}" height="22" rx="3" fill="#4169c8"/>',
                 f'<text x="{left + bar + 8:.3f}" y="{y + 16}" font-size="12">{value:.3f} us</text>']
    body += ['</g></svg>']
    Path(path).write_text("\n".join(body), encoding="utf-8")
