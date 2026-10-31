"""Download the models and the small sample dataset used by the demo.

* Models: YOLOX Nano / Tiny / S, pre-exported to ONNX by the YOLOX authors
  (Apache-2.0) from the official GitHub release.
* Data: COCO128 (the first 128 images of COCO train2017 with labels) packaged
  by Ultralytics. It is split into a calibration set (for INT8 quantization)
  and an evaluation set. It is downloaded at run time and never committed.

Usage:
    python scripts/download_assets.py            # everything
    python scripts/download_assets.py --models yolox_nano
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
YOLOX = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{}.onnx"
MODELS = {
    "yolox_nano": "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
    "yolox_tiny": "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
    "yolox_s": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
}
COCO128 = ("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
           "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e")
CALIB_COUNT = 32  # first 32 images calibrate, the remaining 96 evaluate


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: str, digest: str) -> None:
    if os.path.exists(dest) and sha256(dest) == digest:
        print(f"ok      {os.path.relpath(dest, ROOT)}")
        return
    print(f"fetch   {url}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest + ".part")
    if sha256(dest + ".part") != digest:
        os.remove(dest + ".part")
        raise SystemExit(f"checksum mismatch for {url}")
    os.replace(dest + ".part", dest)


def get_data() -> None:
    data = os.path.join(ROOT, "data")
    zpath = os.path.join(data, "coco128.zip")
    fetch(COCO128[0], zpath, COCO128[1])
    src = os.path.join(data, "coco128")
    if not os.path.isdir(src):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(data)
    images = sorted(os.listdir(os.path.join(src, "images", "train2017")))
    for split, names in (("calib", images[:CALIB_COUNT]), ("eval", images[CALIB_COUNT:])):
        img_dir = os.path.join(data, split, "images")
        lbl_dir = os.path.join(data, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for n in names:
            shutil.copy(os.path.join(src, "images", "train2017", n), img_dir)
            lbl = os.path.join(src, "labels", "train2017", os.path.splitext(n)[0] + ".txt")
            if os.path.exists(lbl):  # a few COCO images have no objects
                shutil.copy(lbl, lbl_dir)
        print(f"split   data/{split}: {len(names)} images")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--no-data", action="store_true")
    args = ap.parse_args()
    for m in args.models:
        fetch(YOLOX.format(m), os.path.join(ROOT, "models", f"{m}.onnx"), MODELS[m])
    if not args.no_data:
        get_data()


if __name__ == "__main__":
    main()
