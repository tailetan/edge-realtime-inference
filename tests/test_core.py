"""Unit tests that run without any model download."""
import time

import numpy as np
import pytest

from edgeinfer.detector import Detection, decode, letterbox, make_grids, nms, postprocess
from edgeinfer.evaluate import average_precision, compute_map, iou_matrix
from edgeinfer.realtime import FifoQueue, Frame, LatestSlot, run_pipeline


# ------------------------------------------------------------------ pre/post-processing

def test_letterbox_keeps_aspect_and_pads_grey():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    tensor, ratio = letterbox(img, (416, 416))
    assert tensor.shape == (3, 416, 416) and tensor.dtype == np.float32
    assert ratio == pytest.approx(416 / 200)
    assert tensor[:, 300, :].mean() == 114  # bottom padding
    assert tensor[:, 0, 0].tolist() == [0, 0, 0]  # image content


def test_grids_cover_all_strides():
    grids, strides = make_grids((416, 416))
    assert len(grids) == 52 * 52 + 26 * 26 + 13 * 13 == 3549  # matches YOLOX-Nano/Tiny output rows
    assert set(strides.ravel().tolist()) == {8, 16, 32}


def test_decode_maps_cell_offsets_to_pixels():
    grids = np.array([[3, 5]], dtype=np.float32)
    strides = np.array([[8]], dtype=np.float32)
    raw = np.zeros((1, 85), dtype=np.float32)
    raw[0, :2] = 0.5
    out = decode(raw, grids, strides)
    assert out[0, :4].tolist() == [28.0, 44.0, 8.0, 8.0]  # (3.5*8, 5.5*8, e^0*8, e^0*8)


def test_nms_suppresses_overlaps_only():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.5) == [0, 2]


def test_postprocess_is_class_aware():
    # Two identical boxes, different classes: both must survive NMS.
    grids, strides = np.zeros((2, 2), np.float32), np.ones((2, 1), np.float32)
    raw = np.zeros((2, 85), dtype=np.float32)
    raw[:, 0:2], raw[:, 2:4], raw[:, 4] = 50, np.log(20), 1.0
    raw[0, 5 + 0] = 0.9  # person
    raw[1, 5 + 16] = 0.8  # dog
    dets = postprocess(raw, 1.0, grids, strides, score_thr=0.3, iou_thr=0.45)
    assert sorted(d.label for d in dets) == ["dog", "person"]


# ------------------------------------------------------------------ evaluation

def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    assert iou_matrix(a, b)[0].tolist() == pytest.approx([1.0, 1 / 3, 0.0])


def test_ap_perfect_and_empty():
    assert average_precision(np.array([1.0, 1.0]), np.array([0.9, 0.8]), 2) == pytest.approx(1.0)
    assert average_precision(np.array([]), np.array([]), 3) == 0.0


def test_map_perfect_predictions():
    gt = [np.array([[0, 10, 10, 50, 50], [16, 60, 60, 90, 90]], np.float32)]
    preds = [[Detection((10, 10, 50, 50), 0.9, 0), Detection((60, 60, 90, 90), 0.8, 16)]]
    m, m50 = compute_map(preds, gt)
    assert m == pytest.approx(1.0) and m50 == pytest.approx(1.0)


# ------------------------------------------------------------------ real-time buffering

def _frame(i):
    return Frame(i, np.zeros((1, 1, 3), np.uint8), time.perf_counter())


def test_latest_slot_keeps_newest_and_counts_drops():
    slot = LatestSlot()
    for i in range(5):
        slot.put(_frame(i))
    assert slot.get().index == 4
    assert slot.dropped == 4
    slot.close()
    assert slot.get(timeout=0.01) is None and slot.closed_and_empty


def test_fifo_keeps_everything_in_order():
    q = FifoQueue()
    for i in range(3):
        q.put(_frame(i))
    assert [q.get().index for _ in range(3)] == [0, 1, 2]
    assert q.dropped == 0 and q.max_depth == 3


def _slow_process(ms):
    def process(_img):
        time.sleep(ms / 1000)
        return None, ms
    return process


@pytest.mark.parametrize("policy", ["latest", "queue"])
def test_pipeline_processes_everything_when_model_is_fast(policy):
    frames = (np.zeros((4, 4, 3), np.uint8) for _ in range(20))
    stats = run_pipeline(frames, 100.0, _slow_process(1), policy)
    assert stats.captured == 20
    assert stats.processed + stats.dropped == 20


def test_latest_bounds_latency_when_model_is_slow():
    """Camera at 100 FPS, model needs 30 ms: 'latest' drops frames, 'queue' falls behind."""
    def run(policy):
        frames = (np.zeros((4, 4, 3), np.uint8) for _ in range(40))
        return run_pipeline(frames, 100.0, _slow_process(30), policy)

    latest, queue = run("latest"), run("queue")
    assert latest.dropped > 0 and queue.dropped == 0
    assert max(latest.e2e_ms) < 100  # bounded: roughly one inference + one frame
    assert max(queue.e2e_ms) > 3 * max(latest.e2e_ms)  # grows with the backlog


def test_on_result_can_stop_the_run():
    frames = (np.zeros((4, 4, 3), np.uint8) for _ in range(1000))
    seen = []

    def on_result(frame, result, stats):
        seen.append(frame.index)
        return len(seen) < 5

    stats = run_pipeline(frames, 200.0, _slow_process(1), "latest", on_result)
    assert stats.processed == 5
