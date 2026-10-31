"""Produce INT8 versions of a YOLOX ONNX model with ONNX Runtime's quantizer.

Two flavours are written next to the FP32 model:

* ``<name>.int8-dynamic.onnx`` - weights quantized ahead of time, activations
  quantized on the fly at run time. No calibration data needed. Kept for
  comparison: on conv nets it is usually SLOWER than FP32 (see the blog).
* ``<name>.int8-static.onnx``  - weights AND activations quantized ahead of
  time (QDQ format, per-channel weights) using a calibration set to fix the
  activation ranges. This is the variant that actually speeds up conv nets.

Always re-check accuracy afterwards (``python benchmark.py``): post-training
quantization works well for some models and breaks others.

Usage:
    python scripts/quantize.py models/yolox_tiny.onnx
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import onnx
from onnx import version_converter
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from edgeinfer.detector import letterbox  # noqa: E402


class ImageReader(CalibrationDataReader):
    """Feeds letterboxed images to the calibrator, one at a time."""

    def __init__(self, paths: list[str], input_name: str, size: tuple[int, int]):
        self.items = iter(paths)
        self.input_name, self.size = input_name, size

    def get_next(self):
        path = next(self.items, None)
        if path is None:
            return None
        tensor, _ = letterbox(cv2.imread(path), self.size)
        return {self.input_name: tensor[None]}


def prepare(model_path: str, out_path: str) -> onnx.ModelProto:
    """Upgrade to opset 13 if needed and run ORT's shape inference pre-pass.

    The YOLOX release models are opset 11, but per-channel QDQ needs the
    ``axis`` attribute on (De)QuantizeLinear, which arrived in opset 13.
    """
    model = onnx.load(model_path)
    if max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")) < 13:
        model = version_converter.convert_version(model, 13)
    onnx.save(model, out_path)
    quant_pre_process(out_path, out_path, skip_symbolic_shape=True)
    return onnx.load(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="FP32 ONNX model")
    ap.add_argument("--calib", default="data/calib/images", help="folder of calibration images (jpg/png)")
    ap.add_argument("--max-calib", type=int, default=100)
    ap.add_argument("--method", choices=["minmax", "entropy", "percentile"], default="minmax")
    args = ap.parse_args()

    base = os.path.splitext(args.model)[0]
    prep = base + ".prep.onnx"
    model = prepare(args.model, prep)
    inp = model.graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    size = (dims[2], dims[3])

    out_dyn = base + ".int8-dynamic.onnx"
    quantize_dynamic(prep, out_dyn, weight_type=QuantType.QInt8)
    print(f"wrote {out_dyn}")

    paths = sorted(p for ext in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(args.calib, f"*.{ext}")))
    if not paths:
        sys.exit(f"no calibration images in {args.calib} - run `python scripts/download_assets.py` first")
    paths = paths[: args.max_calib]
    method = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
              "percentile": CalibrationMethod.Percentile}[args.method]

    out_static = base + ".int8-static.onnx"
    quantize_static(
        prep, out_static, ImageReader(paths, inp.name, size),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        calibrate_method=method,
    )
    print(f"wrote {out_static} (calibrated on {len(paths)} images, {args.method})")
    os.remove(prep)


if __name__ == "__main__":
    main()
