"""YOLOX object detector running on ONNX Runtime (CPU).

Everything the model needs lives here: letterbox pre-processing, the ONNX
Runtime session, grid decoding of the raw YOLOX head and class-aware NMS.
The code is written with plain NumPy so it runs anywhere ONNX Runtime does.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

from .coco import COCO_CLASSES


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels
    score: float
    class_id: int

    @property
    def label(self) -> str:
        return COCO_CLASSES[self.class_id]


@dataclass
class Timing:
    """Per-frame timing breakdown in milliseconds."""

    pre_ms: float = 0.0
    infer_ms: float = 0.0
    post_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.pre_ms + self.infer_ms + self.post_ms


@dataclass
class SessionConfig:
    # 0 = let ONNX Runtime decide (usually physical cores). On SMT/vCPU machines that can leave
    # cores idle: 59 ms vs 38 ms pinned to 2 threads in results/cloud-vm-2vcpu/benchmark.json.
    intra_threads: int = 0
    inter_threads: int = 1
    graph_opt: str = "all"  # disable | basic | extended | all
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    # OpenVINO only. "f32" for an apples-to-apples comparison with ONNX Runtime.
    # "auto" lets OpenVINO pick: on CPUs with AMX/AVX512-BF16 (recent Xeons) it
    # silently runs FP32 models in bfloat16, which is much faster but no longer FP32.
    ov_precision: str = "f32"


_GRAPH_OPT = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def letterbox(img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Resize keeping aspect ratio and pad bottom/right with grey (114).

    Returns the CHW float32 tensor (no batch dim) and the resize ratio needed
    to map boxes back to the original image.
    """
    h, w = size
    ratio = min(h / img.shape[0], w / img.shape[1])
    new_w, new_h = int(round(img.shape[1] * ratio)), int(round(img.shape[0] * ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((h, w, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), ratio


def make_grids(size: tuple[int, int], strides=(8, 16, 32)) -> tuple[np.ndarray, np.ndarray]:
    """Anchor-free grid centres and strides for every output row of YOLOX."""
    grids, expanded = [], []
    for s in strides:
        hs, ws = size[0] // s, size[1] // s
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(g)
        expanded.append(np.full((g.shape[0], 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(expanded).astype(np.float32)


def decode(raw: np.ndarray, grids: np.ndarray, strides: np.ndarray) -> np.ndarray:
    """Turn raw head output (N, 85) into centre-x, centre-y, w, h in input pixels."""
    out = raw.copy()
    out[:, :2] = (out[:, :2] + grids) * strides
    out[:, 2:4] = np.exp(out[:, 2:4]) * strides
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Plain greedy NMS on xyxy boxes. Returns kept indices, best first."""
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def postprocess(
    raw: np.ndarray,
    ratio: float,
    grids: np.ndarray,
    strides: np.ndarray,
    score_thr: float = 0.3,
    iou_thr: float = 0.45,
) -> list[Detection]:
    pred = decode(raw, grids, strides)
    cls_scores = pred[:, 5:] * pred[:, 4:5]  # objectness x class probability
    class_ids = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(class_ids)), class_ids]
    mask = scores > score_thr
    if not mask.any():
        return []
    pred, scores, class_ids = pred[mask], scores[mask], class_ids[mask]
    boxes = np.empty((len(pred), 4), dtype=np.float32)
    boxes[:, 0] = pred[:, 0] - pred[:, 2] / 2
    boxes[:, 1] = pred[:, 1] - pred[:, 3] / 2
    boxes[:, 2] = pred[:, 0] + pred[:, 2] / 2
    boxes[:, 3] = pred[:, 1] + pred[:, 3] / 2
    boxes /= ratio
    # Class-aware NMS: offset boxes per class so different classes never suppress each other.
    offset = class_ids[:, None].astype(np.float32) * 4096.0
    keep = nms(boxes + offset, scores, iou_thr)
    return [Detection(tuple(map(float, boxes[i])), float(scores[i]), int(class_ids[i])) for i in keep]


class Detector:
    """Load a YOLOX ONNX model once, then call :meth:`detect` per frame."""

    def __init__(self, model_path: str, config: SessionConfig | None = None,
                 score_thr: float = 0.3, iou_thr: float = 0.45, backend: str = "ort"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found - run `python scripts/download_assets.py` first")
        self.config = config or SessionConfig()
        self.backend = backend
        if backend == "ort":
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.config.intra_threads
            opts.inter_op_num_threads = self.config.inter_threads
            opts.graph_optimization_level = _GRAPH_OPT[self.config.graph_opt]
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=self.config.providers)
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            shape = inp.shape
            self._run = lambda t: self.session.run(None, {self.input_name: t})[0]
        elif backend == "openvino":
            # Optional: OpenVINO reads the same .onnx file (FP32 or INT8 QDQ) directly.
            import openvino as ov  # pip install openvino

            ov_cfg = {"PERFORMANCE_HINT": "LATENCY"}
            if self.config.ov_precision != "auto":
                ov_cfg["INFERENCE_PRECISION_HINT"] = self.config.ov_precision
            if self.config.intra_threads:
                ov_cfg["INFERENCE_NUM_THREADS"] = self.config.intra_threads
            compiled = ov.Core().compile_model(model_path, "CPU", ov_cfg)
            request = compiled.create_infer_request()
            shape = list(compiled.input(0).shape)
            self._run = lambda t: request.infer({0: t})[compiled.output(0)]
        else:
            raise ValueError(f"unknown backend {backend!r} (use 'ort' or 'openvino')")
        self.input_size = (int(shape[2]), int(shape[3]))
        self.grids, self.strides = make_grids(self.input_size)
        self.score_thr, self.iou_thr = score_thr, iou_thr
        self.model_path = model_path

    def infer_raw(self, tensor: np.ndarray) -> np.ndarray:
        return self._run(tensor[None])[0]

    def detect(self, img_bgr: np.ndarray) -> tuple[list[Detection], Timing]:
        t0 = time.perf_counter()
        tensor, ratio = letterbox(img_bgr, self.input_size)
        t1 = time.perf_counter()
        raw = self.infer_raw(tensor)
        t2 = time.perf_counter()
        dets = postprocess(raw, ratio, self.grids, self.strides, self.score_thr, self.iou_thr)
        t3 = time.perf_counter()
        return dets, Timing((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)


def draw(img: np.ndarray, dets: list[Detection]) -> np.ndarray:
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.box)
        colour = tuple(int(c) for c in np.random.default_rng(d.class_id).integers(64, 255, 3))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        text = f"{d.label} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out
