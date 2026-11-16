"""P3 groundwork: run a pretrained YOLO over every graded frame and keep every ball and person box.

    uv run python analysis/p3_yolo_pass.py            # resumable; ~2-3 h on an RTX 4060
    uv run python analysis/p3_yolo_pass.py --weights models/ball_v1.pt --out cache/ball_v1_pass \
        --classes 0 --conf 0.1 --fp32                 # the self-trained ball detector (P3), strict fp32

Pretrained YOLO11m (COCO), full frame at imgsz 1920, fp16. Kept per detection: frame, class
(0 person, 32 sports ball), confidence, box centre and size in pixels. Ball boxes are kept from a
low confidence (0.10) so later analyses can choose their own threshold; persons from 0.30.

Output: cache/yolo/chunk_<first>.npz, one file per CHUNK graded frames, so an interrupted run
resumes from the last finished chunk, and manifest.json with the run's settings, the weights'
checksum, the environment and the git commit. These are derived match data: never commit them.

--fp32 is strict fp32 (mtah.gpu.strict_fp32: TF32 off, cuDNN deterministic). Boxes are always
written on the fixed grid of mtah.gpu.quantize_boxes (0.01 px, 1e-4 confidence), and the
confidence threshold is applied on that grid (notes/BULGULAR.md).
"""

import hashlib
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, asset_paths, half_video_blocks  # noqa: E402
from mtah.gpu import quantize_boxes, strict_fp32  # noqa: E402

# Params
OUT = os.path.join(ROOT, "cache", "yolo")
WEIGHTS = os.path.join(ROOT, "models", "yolo11m.pt")
CHUNK = 1000
BATCH = 4
IMGSZ = 1920
CONF = {0: 0.30, 32: 0.10}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--classes", type=int, nargs="+", default=None, help="with --conf, one threshold for all")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--fp32", action="store_true", help="fp32 inference (eda_out/reports/P3_determinism.md)")
    args = ap.parse_args()
    weights, out = os.path.join(ROOT, args.weights) if not os.path.isabs(args.weights) else args.weights, args.out
    out = os.path.join(ROOT, out) if not os.path.isabs(out) else out
    conf_by_cls = {c: args.conf for c in args.classes} if args.classes else CONF
    env = strict_fp32() if args.fp32 else {}
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    LOGGER.setLevel("ERROR")            # one deprecation warning per batch otherwise
    os.makedirs(out, exist_ok=True)
    manifest = dict(env, weights=os.path.relpath(weights, ROOT), weights_sha256=hashlib.sha256(open(weights, "rb").read()).hexdigest(),
                    imgsz=IMGSZ, batch=BATCH, conf=conf_by_cls, fp32=args.fp32,
                    commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    model = YOLO(weights)
    # chunks never straddle the halftime gap: each one is read sequentially from its first frame
    chunks = [np.arange(c, min(c + CHUNK, b + 1)) for _, a, b in half_video_blocks(GAME) for c in range(a, b + 1, CHUNK)]
    total = sum(len(c) for c in chunks)
    cap = cv2.VideoCapture(asset_paths()["video"])
    started, done = time.time(), 0
    for chunk in chunks:
        assert np.all(np.diff(chunk) == 1)
        path = os.path.join(out, "chunk_{:06d}.npz".format(chunk[0]))
        if os.path.exists(path):
            total -= len(chunk)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(chunk[0]))
        rows = []
        for b0 in range(0, len(chunk), BATCH):
            ids = chunk[b0:b0 + BATCH]
            imgs = []
            for _ in ids:
                ok, img = cap.read()
                imgs.append(img if ok else np.zeros((1080, 1920, 3), np.uint8))
            results = model.predict(imgs, imgsz=IMGSZ, conf=min(conf_by_cls.values()), classes=list(conf_by_cls),
                                    half=not args.fp32, verbose=False)
            for f, r in zip(ids, results):
                cls = r.boxes.cls.cpu().numpy().astype(np.int16)
                conf, xywh = quantize_boxes(r.boxes.conf.cpu().numpy(), r.boxes.xywh.cpu().numpy())
                keep = conf >= np.vectorize(conf_by_cls.get)(cls) if len(cls) else np.zeros(0, bool)
                for k in np.flatnonzero(keep):
                    rows.append((f, cls[k], conf[k], *xywh[k]))
        arr = np.array(rows, np.float32).reshape(-1, 7)
        np.savez_compressed(path, frame=arr[:, 0].astype(np.int32), cls=arr[:, 1].astype(np.int16),
                            conf=arr[:, 2], x=arr[:, 3], y=arr[:, 4], w=arr[:, 5], h=arr[:, 6])
        done += len(chunk)
        rate = done / (time.time() - started)
        print("{:6d}/{} frames this run  {:.1f} fps  ETA {:.0f} min".format(
            done, total, rate, (total - done) / rate / 60), flush=True)


if __name__ == "__main__":
    main()
