"""P3 groundwork: colour features of every YOLO person box on sampled frames (OpenCV).

    uv run python analysis/p3b_kit_features.py        # ~20 min, writes cache/kit_features.npz

Every STEP-th graded frame is decoded (sequential grab, retrieve only the sampled frames). For each
YOLO person box (conf >= 0.3) the torso is cut out -- the box's central half in width, 15-50% of its
height from the top -- and grass pixels are masked (OpenCV HSV hue 30-90 with saturation > 50).
Kept per box: frame, box centre/size, foot point, and features of the remaining pixels:
  lab_med (3)   median L*a*b*
  hue_hist (12) hue histogram of saturated pixels (S > 60), normalised
  sv (4)        share of white (S < 40, V > 170), dark (V < 60), saturated pixels, and pixel count
Derived match data: never committed.
"""

import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, asset_paths, half_video_blocks  # noqa: E402
from mtah.keepers import PERSON_CONF, YOLO_DIR  # noqa: E402
from mtah.kits import torso_features  # noqa: E402

# Params
STEP = 25
OUT = os.path.join(ROOT, "cache", "kit_features.npz")


def main():
    import glob
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(YOLO_DIR, "chunk_*.npz")))]
    cols = {c: np.concatenate([p[c] for p in parts]) for c in ("frame", "cls", "conf", "x", "y", "w", "h")}
    keep = (cols["cls"] == 0) & (cols["conf"] >= PERSON_CONF)
    boxes = {c: v[keep] for c, v in cols.items()}
    targets = np.concatenate([np.arange(a, b + 1, STEP) for _, a, b in half_video_blocks(GAME)])
    want = set(targets.tolist())
    order = np.argsort(boxes["frame"], kind="stable")
    bf = boxes["frame"][order]
    cap = cv2.VideoCapture(asset_paths()["video"])
    rows, feats = [], []
    t0, last = time.time(), int(targets.max())
    for f in range(last + 1):
        if not cap.grab():
            break
        if f not in want:
            continue
        ok, img = cap.retrieve()
        if not ok:
            continue
        a, b = np.searchsorted(bf, f), np.searchsorted(bf, f, side="right")
        for i in order[a:b]:
            x, y, w, h = (boxes[c][i] for c in ("x", "y", "w", "h"))
            ft = torso_features(img, x, y, w, h)
            if ft is None:
                continue
            rows.append((f, x, y, w, h, boxes["conf"][i]))
            feats.append(ft)
        if f % (STEP * 400) == 0:
            print("frame {} / {}  {:.0f} s".format(f, last, time.time() - t0), flush=True)
    rows = np.array(rows, float)
    np.savez_compressed(OUT, frame=rows[:, 0].astype(int), x=rows[:, 1], y=rows[:, 2], w=rows[:, 3], h=rows[:, 4],
                        conf=rows[:, 5], feat=np.array(feats, np.float32))
    print("wrote", OUT, len(rows), "boxes from", len(targets), "frames in", int(time.time() - t0), "s")


if __name__ == "__main__":
    main()
