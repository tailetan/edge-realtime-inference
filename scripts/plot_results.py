"""Turn results/<host>/benchmark.json into PNG charts next to it.

Usage:
    python scripts/plot_results.py results/<host>/benchmark.json
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
FRAME_BUDGET_MS = 1000 / 30


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(ax, groups, series, values, fmt):
    n = len(series)
    width = 0.8 / n
    for i, name in enumerate(series):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        ys = values[name]
        bars = ax.bar(xs, [y if y is not None else 0 for y in ys], width * 0.92, color=SERIES[i], label=name)
        for b, y in zip(bars, ys):
            if y is not None:
                ax.annotate(fmt(y), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom",
                            fontsize=8, color=INK, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(groups)), groups, color=INK)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")


def pick(rows, model, precision, runtime, ov=None):
    for r in rows:
        if r["model"] == model and r["precision"] == precision and r["runtime"] == runtime and \
                (ov is None or r.get("ov_precision") == ov):
            return r
    return None


def main(path: str) -> None:
    data = json.load(open(path))
    out = os.path.dirname(path)
    models = ["yolox_nano", "yolox_tiny", "yolox_s"]
    labels = ["Nano", "Tiny", "S"]
    rows = data.get("variants", [])

    if rows:
        series = {"ONNX Runtime FP32": ("fp32", "ort", None), "ONNX Runtime INT8": ("int8-static", "ort", None),
                  "OpenVINO FP32": ("fp32", "openvino", "f32")}
        series = {k: v for k, v in series.items() if any(pick(rows, m, *v) for m in models)}
        lat = {k: [(pick(rows, m, *v) or {}).get("p50_ms") for m in models] for k, v in series.items()}
        fig, ax = plt.subplots(figsize=(8, 4.2))
        style(ax, "Model latency per frame (median)", "milliseconds (lower is better)")
        grouped_bars(ax, labels, list(series), lat, lambda y: f"{y:.0f}")
        ax.axhline(FRAME_BUDGET_MS, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate("30 FPS budget (33 ms)", (-0.45, FRAME_BUDGET_MS), color=MUTED, fontsize=8,
                    xytext=(0, 3), textcoords="offset points", ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "latency.png"), dpi=160)

        acc_series = {"FP32": ("fp32", "ort", None), "INT8 static": ("int8-static", "ort", None),
                      "INT8 dynamic": ("int8-dynamic", "ort", None)}
        acc = {k: [(pick(rows, m, *v) or {}).get("map50_95") for m in models] for k, v in acc_series.items()}
        fig, ax = plt.subplots(figsize=(8, 4.2))
        style(ax, "Accuracy after quantization (ONNX Runtime)", "mAP@0.5:0.95 (higher is better)")
        grouped_bars(ax, labels, list(acc_series), acc, lambda y: f"{y:.2f}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "accuracy.png"), dpi=160)

    rt = data.get("realtime")
    if rt:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        style(ax, "End-to-end delay under a 30 FPS camera (YOLOX-Tiny FP32)", "capture-to-result delay (ms)")
        for i, (policy, label) in enumerate([("queue", "Queue every frame"), ("latest", "Latest frame only")]):
            key = f"yolox_tiny.onnx|{policy}"
            if key in rt:
                ys = rt[key]["e2e_ms_series"]
                ax.plot(range(1, len(ys) + 1), ys, color=SERIES[i], linewidth=2, label=label)
                ax.annotate(label, (len(ys), ys[-1]), color=INK, fontsize=8, xytext=(4, 0),
                            textcoords="offset points", va="center")
        ax.set_xlabel("processed frame #", color=MUTED)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "realtime_policies.png"), dpi=160)

    print(f"charts written to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
