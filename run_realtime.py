"""Run real-time object detection on a webcam, a video file or a synthetic camera.

Examples:
    python run_realtime.py --source 0                                   # webcam, live window
    python run_realtime.py --source clip.mp4 --model models/yolox_tiny.int8-static.onnx
    python run_realtime.py --source synthetic --headless --save out.mp4  # no camera needed
    python run_realtime.py --source synthetic --policy queue --headless  # see latency pile up
"""
from __future__ import annotations

import argparse
import glob
import json

import cv2

from edgeinfer import Detector, SessionConfig, draw
from edgeinfer.realtime import camera_source, image_folder_source, run_pipeline, video_source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="synthetic", help="camera index, video path, or 'synthetic'")
    ap.add_argument("--model", default="models/yolox_nano.onnx")
    ap.add_argument("--policy", choices=["latest", "queue"], default="latest")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of the synthetic camera")
    ap.add_argument("--frames", type=int, default=300, help="length of the synthetic stream")
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = auto)")
    ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--headless", action="store_true", help="no window (servers, CI)")
    ap.add_argument("--save", help="write the annotated stream to this .mp4")
    ap.add_argument("--json", help="write the run summary to this file")
    args = ap.parse_args()

    det = Detector(args.model, SessionConfig(intra_threads=args.threads), score_thr=args.score)

    if args.source == "synthetic":
        paths = sorted(glob.glob("data/eval/images/*.jpg"))[:10] or sorted(glob.glob("assets/*.*"))
        frames, fps = image_folder_source(paths, args.fps, args.frames)
    elif args.source.isdigit():
        frames, fps = camera_source(int(args.source))
    else:
        frames, fps = video_source(args.source)

    writer = None

    def on_result(frame, dets, stats):
        nonlocal writer
        if args.headless and not args.save:
            return True
        out = draw(frame.image, dets)
        e2e = stats.e2e_ms[-1]
        skipped = stats.captured - stats.processed  # dropped (latest) or still waiting (queue)
        hud = f"{stats.policy}  infer {stats.infer_ms[-1]:.0f} ms  e2e {e2e:.0f} ms  not shown {skipped}"
        (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
        cv2.putText(out, hud, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if args.save:
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps or 30.0, (w, h))
            writer.write(out)
        if not args.headless:
            cv2.imshow("edge-realtime-inference (q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    def process(img):
        dets, timing = det.detect(img)
        return dets, timing.infer_ms

    stats = run_pipeline(frames, fps, process, args.policy, on_result)
    if writer is not None:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    summary = {"model": args.model, **stats.summary()}
    print(json.dumps(summary, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
