"""edgeinfer: real-time object detection on a CPU with ONNX Runtime."""

from .detector import Detection, Detector, SessionConfig, Timing, draw

__all__ = ["Detection", "Detector", "SessionConfig", "Timing", "draw"]
__version__ = "0.1.0"
