"""Integration tests against the real models. Skipped until assets are downloaded."""
import os

import cv2
import pytest

from edgeinfer import Detector

ROOT = os.path.join(os.path.dirname(__file__), "..")
NANO = os.path.join(ROOT, "models", "yolox_nano.onnx")
DOG = os.path.join(ROOT, "assets", "dog.jpg")

needs_model = pytest.mark.skipif(not os.path.exists(NANO), reason="run scripts/download_assets.py first")


@needs_model
def test_nano_finds_the_dog_bicycle_and_car():
    dets, timing = Detector(NANO).detect(cv2.imread(DOG))
    labels = {d.label for d in dets}
    assert {"dog", "bicycle", "car"} <= labels
    assert timing.total_ms > 0
    dog = next(d for d in dets if d.label == "dog")
    x1, y1, x2, y2 = dog.box
    assert 100 < x1 < 160 and 180 < y1 < 240 and 290 < x2 < 360 and 500 < y2 < 576  # dog is lower-left


@needs_model
def test_missing_model_has_helpful_error():
    with pytest.raises(FileNotFoundError, match="download_assets"):
        Detector("models/does-not-exist.onnx")
