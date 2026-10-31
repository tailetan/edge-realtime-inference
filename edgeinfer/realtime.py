"""Real-time frame pipeline: a capture thread feeding an inference loop.

A camera produces frames at its own pace whether or not the model is ready.
When the model is slower than the camera there are two ways to cope:

* ``queue``  - buffer every frame and process them in order. Nothing is lost,
  but each frame waits behind all the earlier ones, so the delay between
  "something happened" and "the system reacted" keeps growing.
* ``latest`` - keep only the newest frame in a one-slot mailbox. Frames that
  arrive while the model is busy are dropped, but the frame being processed
  is always fresh, so end-to-end latency stays bounded.

Real-time systems almost always want ``latest``. The ``queue`` policy is here
so the difference can be measured, not because it is a good idea.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    image: np.ndarray
    captured_at: float  # time.perf_counter() when the frame "hit the sensor"


class LatestSlot:
    """One-slot mailbox. ``put`` overwrites; overwritten frames count as dropped."""

    def __init__(self):
        self._frame: Frame | None = None
        self._cond = threading.Condition()
        self._closed = False
        self.dropped = 0

    def put(self, frame: Frame) -> None:
        with self._cond:
            if self._frame is not None:
                self.dropped += 1
            self._frame = frame
            self._cond.notify()

    def get(self, timeout: float = 1.0) -> Frame | None:
        with self._cond:
            if self._frame is None and not self._closed:
                self._cond.wait(timeout)
            frame, self._frame = self._frame, None
            return frame

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed_and_empty(self) -> bool:
        with self._cond:
            return self._closed and self._frame is None


class FifoQueue:
    """Unbounded FIFO: never drops, so latency grows when the consumer is slow."""

    def __init__(self):
        self._q: collections.deque[Frame] = collections.deque()
        self._cond = threading.Condition()
        self._closed = False
        self.dropped = 0
        self.max_depth = 0

    def put(self, frame: Frame) -> None:
        with self._cond:
            self._q.append(frame)
            self.max_depth = max(self.max_depth, len(self._q))
            self._cond.notify()

    def get(self, timeout: float = 1.0) -> Frame | None:
        with self._cond:
            if not self._q and not self._closed:
                self._cond.wait(timeout)
            return self._q.popleft() if self._q else None

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed_and_empty(self) -> bool:
        with self._cond:
            return self._closed and not self._q


# --------------------------------------------------------------------------- sources

def camera_source(index: int) -> tuple[Iterator[np.ndarray], float | None]:
    """A real webcam. The driver paces frames, so no extra pacing is applied."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {index}")

    def gen():
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    return
                yield img
        finally:
            cap.release()
    return gen(), None


def video_source(path: str) -> tuple[Iterator[np.ndarray], float]:
    """A video file, replayed at its native frame rate as if it were a live camera."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def gen():
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    return
                yield img
        finally:
            cap.release()
    return gen(), fps


def image_folder_source(paths: list[str], fps: float, frames: int,
                        size: tuple[int, int] = (1280, 720)) -> tuple[Iterator[np.ndarray], float]:
    """A synthetic 'camera' that slowly pans across still images.

    Useful on machines without a webcam (CI, servers) and for repeatable
    benchmarks: every run sees exactly the same frames at the same rate.
    """
    images = [cv2.imread(p) for p in paths]
    images = [i for i in images if i is not None]
    if not images:
        raise RuntimeError("no readable images for the synthetic source")
    w, h = size

    def gen():
        for n in range(frames):
            img = images[(n // 30) % len(images)]  # switch image every 30 frames
            scale = max(w / img.shape[1], h / img.shape[0]) * 1.15
            big = cv2.resize(img, None, fx=scale, fy=scale)
            t = (n % 30) / 29
            x = int((big.shape[1] - w) * t)
            y = int((big.shape[0] - h) * 0.5)
            yield np.ascontiguousarray(big[y:y + h, x:x + w])
    return gen(), fps


# --------------------------------------------------------------------------- pipeline

@dataclass
class RunStats:
    policy: str
    source_fps: float | None
    captured: int = 0
    processed: int = 0
    dropped: int = 0
    max_queue_depth: int = 0
    wall_s: float = 0.0
    e2e_ms: list[float] = field(default_factory=list)  # capture -> result available
    infer_ms: list[float] = field(default_factory=list)  # model call only

    def summary(self) -> dict:
        e2e = np.array(self.e2e_ms) if self.e2e_ms else np.array([np.nan])
        inf = np.array(self.infer_ms) if self.infer_ms else np.array([np.nan])
        return {
            "policy": self.policy,
            "source_fps": self.source_fps,
            "captured": self.captured,
            "processed": self.processed,
            "dropped": self.dropped,
            "drop_rate": round(self.dropped / max(self.captured, 1), 3),
            "max_queue_depth": self.max_queue_depth,
            "processed_fps": round(self.processed / max(self.wall_s, 1e-9), 2),
            "infer_ms_p50": round(float(np.percentile(inf, 50)), 2),
            "e2e_ms_p50": round(float(np.percentile(e2e, 50)), 1),
            "e2e_ms_p95": round(float(np.percentile(e2e, 95)), 1),
            "e2e_ms_max": round(float(e2e.max()), 1),
            "e2e_ms_first10_mean": round(float(np.mean(e2e[:10])), 1),
            "e2e_ms_last10_mean": round(float(np.mean(e2e[-10:])), 1),
        }


def run_pipeline(
    frames: Iterator[np.ndarray],
    source_fps: float | None,
    process: Callable[[np.ndarray], tuple[object, float]],
    policy: str = "latest",
    on_result: Callable[[Frame, object, RunStats], bool] | None = None,
) -> RunStats:
    """Run capture and inference concurrently.

    ``process(image) -> (result, infer_ms)`` is the model call.
    ``on_result(frame, result, stats)`` is called after every inference (for
    drawing / saving); returning False stops the run early.
    If ``source_fps`` is set, the capture thread paces itself to that rate,
    imitating a camera sensor that does not wait for anyone.
    """
    buf = LatestSlot() if policy == "latest" else FifoQueue()
    stats = RunStats(policy=policy, source_fps=source_fps)
    stop = threading.Event()

    def capture():
        period = 1.0 / source_fps if source_fps else 0.0
        start = time.perf_counter()
        for i, img in enumerate(frames):
            if stop.is_set():
                break
            if period:  # wait until this frame's slot in the camera timeline
                delay = start + i * period - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            buf.put(Frame(i, img, time.perf_counter()))
            stats.captured += 1
        buf.close()

    t = threading.Thread(target=capture, name="capture", daemon=True)
    t0 = time.perf_counter()
    t.start()
    while True:
        frame = buf.get()
        if frame is None:
            if buf.closed_and_empty:
                break
            continue
        result, infer_ms = process(frame.image)
        stats.e2e_ms.append((time.perf_counter() - frame.captured_at) * 1e3)
        stats.infer_ms.append(infer_ms)
        stats.processed += 1
        if on_result is not None and on_result(frame, result, stats) is False:
            stop.set()
            break
    stats.wall_s = time.perf_counter() - t0
    stop.set()
    t.join(timeout=2)
    stats.dropped = buf.dropped
    stats.max_queue_depth = getattr(buf, "max_depth", 0)
    return stats
