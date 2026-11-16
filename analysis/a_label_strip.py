"""The tactical video with our labels as a scrolling strip underneath, for watching (debugging only).

    uv run python analysis/a_label_strip.py [START_MIN END_MIN] [--sub evaluation/results/v3_chain.csv]
        # -> eda_out/video/label_strip_<start>_<end>.mp4   (local only: cache/eda_out are gitignored,
        #    and the terms forbid publishing clipped video)

Watching the video to understand errors is allowed (RULES.md); turning what is seen into labels,
corrections, or rules and constants tied to specific frames is not. Use it to find KINDS of error.

Under the picture, a strip SPAN seconds wide scrolls right to left; the white line in the middle is
the current frame. Rows: the submission (default v3), the v1 chain, whether the ATD sees the ball, and
Smart Tagging's side. Colours: H red (home, Barcelona), A blue (away, Athletic), D grey, X black.
On the picture, the ATD ball is a yellow ring. Top-left: match clock, frame, the labels now.
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import asset_paths  # noqa: E402
from mtah.data import load_tactical  # noqa: E402
from mtah.frames import load_frames  # noqa: E402

OUT_DIR = os.path.join(ROOT, "eda_out", "video")
W_OUT, H_VID = 1280, 720
ROW_H, SPAN = 22, 30                    # px per strip row, seconds shown across the strip
COL = {pm.H: (40, 40, 220), pm.A: (220, 140, 40), pm.D: (150, 150, 150), pm.X: (0, 0, 0)}   # BGR


def codes(path):
    return pm.label_codes(pm.read_label_csv(path))


def strip_rows(sub):
    fr = load_frames()
    m = load_tactical()
    seen = np.where(np.isnan(m["bx"]), pm.X, pm.D).astype(np.int8)
    tag = np.where(fr["tag_H"].values > fr["tag_A"].values, pm.H, np.where(fr["tag_A"].values > fr["tag_H"].values, pm.A, pm.X)).astype(np.int8)
    v1 = os.path.join(ROOT, "evaluation", "results", "v1_chain.csv")
    rows = [(os.path.basename(sub).replace("_chain.csv", ""), codes(sub))]
    if os.path.exists(v1):
        rows.append(("v1", codes(v1)))
    rows += [("ATD top", seen), ("Smart T.", tag)]
    return rows, m


def render_strip(rows, f, n):
    half = SPAN * 25 // 2
    idx = np.clip(np.linspace(f - half, f + half, W_OUT).astype(int), 0, n - 1)
    out = np.zeros((ROW_H * len(rows), W_OUT, 3), np.uint8)
    for r, (name, c) in enumerate(rows):
        band = np.array([COL[int(v)] for v in c[idx]], np.uint8)
        out[r * ROW_H:(r + 1) * ROW_H - 2] = band[None]
        cv2.putText(out, name, (4, r * ROW_H + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    out[:, W_OUT // 2 - 1:W_OUT // 2 + 1] = 255
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start_min", nargs="?", type=float, default=0.0)
    ap.add_argument("end_min", nargs="?", type=float, default=None)
    ap.add_argument("--sub", default=os.path.join(ROOT, "evaluation", "results", "v3_chain.csv"))
    a = ap.parse_args()
    rows, m = strip_rows(a.sub)
    n = len(rows[0][1])
    first = int(a.start_min * 60 * 25)
    last = n if a.end_min is None else min(n, int(a.end_min * 60 * 25))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "label_strip_{:g}_{:g}.mp4".format(a.start_min, a.end_min if a.end_min is not None else last / 1500))
    cap = cv2.VideoCapture(asset_paths()["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    H_out = H_VID + ROW_H * len(rows)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W_OUT, H_out))
    sx, sy = W_OUT / 1920, H_VID / 1080
    hom = m["hom"]
    for f in range(first, last):
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (W_OUT, H_VID), interpolation=cv2.INTER_AREA)
        if not np.isnan(m["bx"][f]):
            p = hom[f] @ np.array([m["bx"][f], m["by"][f], 1.0])
            if p[2] > 0:
                cv2.circle(img, (int(p[0] / p[2] * sx), int(p[1] / p[2] * sy)), 9, (0, 255, 255), 2, cv2.LINE_AA)
        now = " ".join("{}:{}".format(name, pm.LABELS[int(c[f])]) for name, c in rows[:2])
        cv2.putText(img, "{:02d}:{:02d}  frame {}  {}".format(f // 1500, (f // 25) % 60, f, now), (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(np.vstack([img, render_strip(rows, f, n)]))
        if (f - first) % 7500 == 0:
            print(f, "/", last, flush=True)
    vw.release()
    print("wrote", out)


if __name__ == "__main__":
    main()
