"""Reproducible benchmark for CPU real-time object detection.

Runs five experiments and writes everything to results/<hostname>/:

  1. variants  - every model x precision x runtime: latency, size, accuracy
  2. threads   - how latency scales with intra-op threads
  3. graphopt  - what ONNX Runtime's graph optimizer is worth
  4. breakdown - where a frame's time goes (pre-process / model / post-process)
  5. realtime  - "latest frame" vs "queue every frame" under a 30 FPS camera

Usage:
    python benchmark.py                       # everything (~10-20 min on a laptop)
    python benchmark.py --only variants realtime
    python benchmark.py --quick               # fewer runs, for a smoke test

Numbers are only meaningful for the machine they were measured on. Run this
on the hardware you plan to deploy to.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import socket
import statistics
import time

import cv2
import numpy as np
import onnxruntime as ort

from edgeinfer import Detector, SessionConfig
from edgeinfer.detector import letterbox
from edgeinfer.evaluate import evaluate
from edgeinfer.realtime import image_folder_source, run_pipeline

MODELS = ["yolox_nano", "yolox_tiny", "yolox_s"]
PRECISIONS = {"fp32": "", "int8-dynamic": ".int8-dynamic", "int8-static": ".int8-static"}


def system_info() -> dict:
    info = {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "onnxruntime": ort.__version__, "cpu": platform.processor(), "logical_cpus": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
        info["cpu"] = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("model name"))
        flags = set(next(line for line in text.splitlines() if line.startswith("flags")).split())
        info["isa"] = sorted(f for f in flags if f.startswith(("avx", "amx")) or f in ("fma", "f16c"))
    except (OSError, StopIteration):
        pass
    try:
        import openvino as ov
        info["openvino"] = ov.__version__
    except ImportError:
        info["openvino"] = None
    return info


def latency(det: Detector, img: np.ndarray, warmup: int, runs: int) -> dict:
    """Time the model call alone on a fixed input (pre/post excluded)."""
    tensor, _ = letterbox(img, det.input_size)
    for _ in range(warmup):
        det.infer_raw(tensor)
    samples = []
    for _ in range(runs):
        t = time.perf_counter()
        det.infer_raw(tensor)
        samples.append((time.perf_counter() - t) * 1e3)
    return {"p50_ms": round(float(np.percentile(samples, 50)), 2),
            "p95_ms": round(float(np.percentile(samples, 95)), 2),
            "mean_ms": round(statistics.fmean(samples), 2)}


def exp_variants(args, img) -> list[dict]:
    rows = []
    backends = [("ort", "n/a")]
    if args.openvino:
        backends += [("openvino", "f32"), ("openvino", "auto")]
    for model in MODELS:
        for prec, suffix in PRECISIONS.items():
            path = f"models/{model}{suffix}.onnx"
            if not os.path.exists(path):
                print(f"  skip {path} (missing - run scripts/quantize.py)")
                continue
            for backend, ov_prec in backends:
                if backend == "openvino" and prec == "int8-dynamic":
                    continue  # OpenVINO does not execute ORT's dynamic-quant ops efficiently; not a fair row
                if backend == "openvino" and prec != "fp32" and ov_prec == "auto":
                    continue  # 'auto' only changes how FP32 graphs run
                cfg = SessionConfig(intra_threads=args.threads, ov_precision=ov_prec if backend == "openvino" else "f32")
                det = Detector(path, cfg, backend=backend)
                lat = latency(det, img, args.warmup, args.runs)
                acc_det = Detector(path, cfg, score_thr=0.01, iou_thr=0.65, backend=backend)
                acc = evaluate(acc_det, "data/eval", limit=args.eval_limit)
                row = {"model": model, "precision": prec, "runtime": backend,
                       "ov_precision": ov_prec if backend == "openvino" else None,
                       "size_mb": round(os.path.getsize(path) / 1e6, 2), **lat,
                       "map50_95": round(acc.map50_95, 3), "map50": round(acc.map50, 3), "eval_images": acc.images}
                print("  ", row)
                rows.append(row)
    return rows


def exp_threads(args, img) -> list[dict]:
    rows = []
    max_t = os.cpu_count() or 1
    counts = [0] + sorted({1, 2, 4, 8, max_t} & set(range(1, max_t + 1)))  # 0 = runtime default
    for model in ["yolox_tiny.onnx", "yolox_tiny.int8-static.onnx"]:
        for n in counts:
            det = Detector(f"models/{model}", SessionConfig(intra_threads=n))
            row = {"model": model, "threads": n, **latency(det, img, args.warmup, args.runs)}
            print("  ", row)
            rows.append(row)
    return rows


def exp_graphopt(args, img) -> list[dict]:
    rows = []
    for level in ["disable", "basic", "extended", "all"]:
        det = Detector("models/yolox_tiny.onnx", SessionConfig(intra_threads=args.threads, graph_opt=level))
        row = {"model": "yolox_tiny.onnx", "graph_opt": level, **latency(det, img, args.warmup, args.runs)}
        print("  ", row)
        rows.append(row)
    return rows


def exp_breakdown(args, frames: list[np.ndarray]) -> list[dict]:
    rows = []
    for model in ["yolox_nano.onnx", "yolox_tiny.onnx", "yolox_tiny.int8-static.onnx"]:
        det = Detector(f"models/{model}", SessionConfig(intra_threads=args.threads), score_thr=0.35)
        for f in frames[: args.warmup]:
            det.detect(f)
        pre, inf, post = [], [], []
        for i in range(args.runs):
            _, t = det.detect(frames[i % len(frames)])
            pre.append(t.pre_ms); inf.append(t.infer_ms); post.append(t.post_ms)
        row = {"model": model, "pre_ms": round(float(np.median(pre)), 2),
               "infer_ms": round(float(np.median(inf)), 2), "post_ms": round(float(np.median(post)), 2)}
        print("  ", row)
        rows.append(row)
    return rows


def exp_realtime(args) -> dict:
    out = {}
    paths = sorted(glob.glob("data/eval/images/*.jpg"))[:10]
    for model in ["yolox_tiny.onnx", "yolox_tiny.int8-static.onnx"]:
        det = Detector(f"models/{model}", SessionConfig(intra_threads=args.threads), score_thr=0.35)
        det.detect(cv2.imread(paths[0]))  # warm up

        def process(image, det=det):
            dets, timing = det.detect(image)
            return dets, timing.infer_ms

        for policy in ["latest", "queue"]:
            frames, fps = image_folder_source(paths, 30.0, args.rt_frames)
            stats = run_pipeline(frames, fps, process, policy)
            key = f"{model}|{policy}"
            out[key] = {"summary": {"model": model, **stats.summary()},
                        "e2e_ms_series": [round(x, 1) for x in stats.e2e_ms]}
            print("  ", out[key]["summary"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=["variants", "threads", "graphopt", "breakdown", "realtime"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-openvino", dest="openvino", action="store_false")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                    help="intra-op threads for every runtime (default: all logical CPUs). Pinned so that "
                         "ONNX Runtime and OpenVINO are compared on equal terms - their defaults differ.")
    args = ap.parse_args()
    args.warmup, args.runs = (5, 20) if args.quick else (10, 100)
    args.eval_limit = 16 if args.quick else None
    args.rt_frames = 90 if args.quick else 300
    if args.openvino:
        try:
            import openvino  # noqa: F401
        except ImportError:
            print("openvino not installed - skipping OpenVINO rows (pip install openvino)")
            args.openvino = False

    out_dir = args.out or os.path.join("results", socket.gethostname())
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(sorted(glob.glob("data/eval/images/*.jpg"))[0])
    frames = [cv2.imread(p) for p in sorted(glob.glob("data/eval/images/*.jpg"))[:20]]

    results = {"system": system_info(), "settings": {k: v for k, v in vars(args).items() if k != "only"}}
    print(json.dumps(results["system"], indent=2))
    for name in args.only:
        print(f"\n== {name}")
        if name == "variants":
            results[name] = exp_variants(args, img)
        elif name == "threads":
            results[name] = exp_threads(args, img)
        elif name == "graphopt":
            results[name] = exp_graphopt(args, img)
        elif name == "breakdown":
            results[name] = exp_breakdown(args, frames)
        elif name == "realtime":
            results[name] = exp_realtime(args)

    path = os.path.join(out_dir, "benchmark.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {path}  ->  python scripts/plot_results.py {path}")


if __name__ == "__main__":
    main()
