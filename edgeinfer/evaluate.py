"""Small, dependency-free COCO-style mAP evaluator for YOLO-format labels.

It follows the COCO recipe (greedy matching by score, 101-point interpolated
precision, IoU thresholds 0.50:0.05:0.95) closely enough to compare FP32 and
INT8 variants of the same model. It is NOT a drop-in replacement for
pycocotools: use it for relative comparisons, not leaderboard numbers.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection, Detector

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


@dataclass
class EvalResult:
    map50_95: float
    map50: float
    images: int
    mean_infer_ms: float


def load_labels(path: str, width: int, height: int) -> np.ndarray:
    """YOLO txt (class cx cy w h, normalised) -> array of (class, x1, y1, x2, y2) in pixels."""
    if not os.path.exists(path):
        return np.zeros((0, 5), dtype=np.float32)
    rows = np.loadtxt(path, ndmin=2, dtype=np.float32)
    cls, cx, cy, w, h = rows.T
    return np.stack([cls, (cx - w / 2) * width, (cy - h / 2) * height,
                     (cx + w / 2) * width, (cy + h / 2) * height], 1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP for one class at one IoU threshold."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # make monotonic
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def compute_map(preds: list[list[Detection]], gts: list[np.ndarray], num_classes: int = 80) -> tuple[float, float]:
    """Return (mAP@0.5:0.95, mAP@0.5) over all classes that appear in the ground truth."""
    # Per class, per IoU threshold: list of (score, is_tp)
    records = {c: ([], [[] for _ in IOU_THRESHOLDS]) for c in range(num_classes)}
    n_gt = np.zeros(num_classes, dtype=int)
    for dets, gt in zip(preds, gts):
        for c in range(num_classes):
            g = gt[gt[:, 0] == c, 1:]
            n_gt[c] += len(g)
            d = [x for x in dets if x.class_id == c]
            if not d:
                continue
            d.sort(key=lambda x: -x.score)
            boxes = np.array([x.box for x in d], dtype=np.float32)
            ious = iou_matrix(boxes, g)
            scores, tps = records[c]
            scores.extend(x.score for x in d)
            for t, thr in enumerate(IOU_THRESHOLDS):
                taken = np.zeros(len(g), dtype=bool)
                for i in range(len(d)):
                    hit = 0
                    if len(g):
                        cand = np.where(~taken & (ious[i] >= thr))[0]
                        if len(cand):
                            j = cand[np.argmax(ious[i, cand])]
                            taken[j] = True
                            hit = 1
                    tps[t].append(hit)
    ap = np.full((num_classes, len(IOU_THRESHOLDS)), np.nan)
    for c in range(num_classes):
        if n_gt[c] == 0:
            continue
        scores, tps = records[c]
        s = np.array(scores, dtype=np.float32)
        for t in range(len(IOU_THRESHOLDS)):
            ap[c, t] = average_precision(np.array(tps[t], dtype=np.float32), s, n_gt[c])
    return float(np.nanmean(ap)), float(np.nanmean(ap[:, 0]))


def evaluate(detector: Detector, split_dir: str, limit: int | None = None) -> EvalResult:
    paths = sorted(glob.glob(os.path.join(split_dir, "images", "*.jpg")))[:limit]
    preds, gts, infer = [], [], []
    for p in paths:
        img = cv2.imread(p)
        dets, timing = detector.detect(img)
        preds.append(dets)
        infer.append(timing.infer_ms)
        lbl = os.path.join(split_dir, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
        gts.append(load_labels(lbl, img.shape[1], img.shape[0]))
    m, m50 = compute_map(preds, gts)
    return EvalResult(m, m50, len(paths), float(np.mean(infer)))
