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
    root = Path(__file__).resolve().parent
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(), "torch": torch.__version__,
        "os": platform.platform(), "processor": platform.processor(),
        "device": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else "CPU",
        "torch_threads": torch.get_num_threads(), "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": {str(Path(p).relative_to(root)).replace("\\", "/"):
                           hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        "torch_build": torch.__config__.show(),
    }


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def samples(fn, device="cpu", count=9, iterations=10, warmup=3):
    for _ in range(warmup):
        fn()
    synchronize(device)
    values = []
    for _ in range(count):
        synchronize(device)
        start = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        synchronize(device)
        values.append((time.perf_counter_ns() - start) / 1000 / iterations)
    return values


def stats(values):
    return {"samples_us": values, "median_us": statistics.median(values),
            "min_us": min(values), "max_us": max(values)}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def bar_svg(path, rows, title, subtitle):
    """Simple exported measurement plot. Values must be measured, not hypotheses."""
    width, left, right, top, line = 1000, 360, 110, 95, 34
    height = top + len(rows) * line + 45
    maximum = max((value for _, value in rows), default=1) or 1
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f7f9fc"/>',
            '<g font-family="Arial, sans-serif" fill="#172b4d">',
            f'<text x="24" y="32" font-size="22">{html.escape(title)}</text>',
            f'<text x="24" y="59" font-size="13">{html.escape(subtitle)}</text>']
    for index, (label, value) in enumerate(rows):
        y = top + index * line
        bar = value / maximum * (width - left - right)
        body += [f'<text x="24" y="{y + 16}" font-size="13">{html.escape(label)}</text>',
                 f'<rect x="{left}" y="{y}" width="{bar:.3f}" height="22" rx="3" fill="#4169c8"/>',
                 f'<text x="{left + bar + 8:.3f}" y="{y + 16}" font-size="12">{value:.3f} us</text>']
    body += ['</g></svg>']
    Path(path).write_text("\n".join(body), encoding="utf-8")
