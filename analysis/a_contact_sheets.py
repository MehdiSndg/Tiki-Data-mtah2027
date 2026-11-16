"""EDA section A: one image per ground-truth transition, six crops around the ball, tracking drawn on.

    uv run python analysis/a_contact_sheets.py [--offsets -1 -0.4 0 0.4 1 2]

Each crop is centred on the ATD ball (or, if it is missing, the last/next place it was seen, else
the image centre), at native resolution so a few-pixel ball stays visible. Home slots are drawn
blue, away orange, goalkeepers with a ring; the ATD ball as a white ring. The ground-truth label
of each crop's frame is printed on it.

Output goes to eda_out/A/sheets/ -- these are video frames, never commit them.
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from a_gt_transitions import LABELS, WINDOW, segments  # noqa: E402
from games import asset_paths  # noqa: E402
from mtah.data import FPS, GK_SLOTS, load_tactical, project  # noqa: E402

# Params
OUT = os.path.join(ROOT, "eda_out", "A", "sheets")
CROP_W, CROP_H = 640, 360
COLS = 3
BGR = {0: (214, 120, 42), 1: (52, 104, 235)}          # home #2a78d6, away #eb6834, as BGR


def anchor(m, f):
    """Pitch point to centre the crop on: the ball, else the nearest-in-time ball within 2 s."""
    for delta in range(0, 2 * FPS):
        for g in (f - delta, f + delta):
            if 0 <= g < len(m["bx"]) and not np.isnan(m["bx"][g]):
                return m["bx"][g], m["by"][g], delta == 0
    return None


def crop(img, m, f, gt_label):
    out = img.copy()
    hom = m["hom"][f]
    for s in np.flatnonzero(~np.isnan(m["px"][f])):
        x, y = project(hom, m["px"][f, s], m["py"][f, s])
        team = int(m["slot_team"][s])
        cv2.circle(out, (int(x), int(y)), 5, BGR[team], -1)
        if s == GK_SLOTS[team]:
            cv2.circle(out, (int(x), int(y)), 10, BGR[team], 2)
    a = anchor(m, f)
    if a is not None and a[2]:
        x, y = project(hom, a[0], a[1])
        cv2.circle(out, (int(x), int(y)), 13, (255, 255, 255), 2)
    if a is not None:
        cx, cy = project(hom, a[0], a[1])
    else:
        cx, cy = img.shape[1] / 2, img.shape[0] / 2
    x0 = int(np.clip(cx - CROP_W / 2, 0, img.shape[1] - CROP_W))
    y0 = int(np.clip(cy - CROP_H / 2, 0, img.shape[0] - CROP_H))
    c = out[y0:y0 + CROP_H, x0:x0 + CROP_W].copy()
    return c, (a is not None and a[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offsets", type=float, nargs="+", default=[-1.0, -0.4, 0.0, 0.4, 1.0, 2.0],
                        help="seconds relative to the transition frame")
    parser.add_argument("--only", type=int, nargs="*", help="transition indices to render")
    parser.add_argument("--cols", type=int, default=COLS, help="tiles per row")
    parser.add_argument("--tag", default="", help="suffix for the output file names")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    m = load_tactical()
    gt = m["gt_public"]
    segs = segments(gt, *WINDOW)
    cap = cv2.VideoCapture(asset_paths()["video"])

    for i in range(1, len(segs)):
        if args.only and i not in args.only:
            continue
        f0 = segs[i][0]
        title = "#{} frame {}  {}->{}".format(i, f0, LABELS[segs[i - 1][2]], LABELS[segs[i][2]])
        tiles = []
        for off in args.offsets:
            f = int(round(f0 + off * FPS))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                img = np.zeros((1080, 1920, 3), np.uint8)
            tile, has_ball = crop(img, m, f, LABELS[gt[f]])
            text = "{:+.1f}s f{} GT={}{}".format(off, f, LABELS[gt[f]], "" if has_ball else " (no ATD ball)")
            cv2.rectangle(tile, (0, 0), (CROP_W, 26), (0, 0, 0), -1)
            cv2.putText(tile, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows = [np.hstack(tiles[r:r + args.cols]) for r in range(0, len(tiles), args.cols)]
        sheet = np.vstack(rows)
        bar = np.zeros((30, sheet.shape[1], 3), np.uint8)
        cv2.putText(bar, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        path = os.path.join(OUT, "t{:02d}_{}{}.jpg".format(i, f0, args.tag))
        cv2.imwrite(path, np.vstack([bar, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 82])
        print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
