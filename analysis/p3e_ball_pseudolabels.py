"""P3 Task 3a: a ball training set from our own match, labelled by agreement, not by a person.

    uv run python analysis/p3e_ball_pseudolabels.py      # ~20 min; cache/ball_v1/ (gitignored)

A positive is a pretrained-YOLO ball box (conf >= 0.3) that the ATD confirms: on the pitch, not on a
painted spot, within AGREE_PX of the projected ATD ball, ATD ball speed below SLOW (a fast ball may
be in the air, where the image ball sits above the ATD's ground point). Tiles of TILE x TILE pixels at
native resolution around each positive (random offset). Hard negatives: tiles on a penalty or centre
spot while the ATD ball is far, and random tiles away from every ball box.

The public window (and a 10 s margin) is left out. Frames are split by 10-minute blocks: every
fifth block is validation. At most PER_WINDOW frames per 10 s, spread in time.
Writes a YOLO dataset: images/{train,val}, labels/{train,val}, data.yaml.
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import asset_paths  # noqa: E402
from mtah.ball import OBSERVED  # noqa: E402
from mtah.data import PITCH, load_tactical  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from p3a_yolo_ball import SPOTS, atd_ball_px, load_yolo, on_pitch, on_spot  # noqa: E402

# Params
OUT = os.path.join(ROOT, "cache", "ball_v1")
AGREE_PX = 30
SLOW = 8.0
TILE = 640
PER_WINDOW, WINDOW = 5, 250
PUBLIC = (11185 - 250, 18685 + 250)
BLOCK = 10 * 60 * 25
N_SPOT_NEG, N_RANDOM_NEG = 600, 600
SEED = 0


def split_of(f):
    return "val" if (f // BLOCK) % 5 == 4 else "train"


def tile_origin(rng, u, v):
    x0 = int(np.clip(u - rng.uniform(0.15, 0.85) * TILE, 0, 1920 - TILE))
    y0 = int(np.clip(v - rng.uniform(0.15, 0.85) * TILE, 0, 1080 - TILE))
    return x0, y0


def main():
    rng = np.random.RandomState(SEED)
    m, df = load_tactical(), load_frames()
    d, covered = load_yolo()
    balls = d[d.cls == 32].copy()
    balls["pitch"] = on_pitch(balls, m["hom"])
    balls["spot"] = on_spot(balls, m["hom"])
    u, v = atd_ball_px(m)
    f = balls.frame.values
    balls["atd_d"] = np.hypot(balls.x.values - u[f], balls.y.values - v[f])
    ok_atd = (df["ball_status"].values == OBSERVED) & (df["ball_speed"].values < SLOW)
    pos = balls[(balls.conf >= 0.3) & balls.pitch & ~balls.spot & (balls.atd_d < AGREE_PX) & ok_atd[f]]
    pos = pos[(pos.frame < PUBLIC[0]) | (pos.frame > PUBLIC[1])]
    pos = pos.sort_values("conf").groupby("frame").tail(1)
    chosen = []
    for w, g in pos.groupby(pos.frame // WINDOW):
        idx = np.linspace(0, len(g) - 1, min(PER_WINDOW, len(g))).round().astype(int)
        chosen.append(g.iloc[np.unique(idx)])
    pos = __import__("pandas").concat(chosen)
    # hard negatives: spot in view, ATD ball > 15 m away from it
    neg = []
    gframes = covered[(covered < PUBLIC[0]) | (covered > PUBLIC[1])]
    cand = rng.choice(gframes, 20000, replace=False)
    for fr in cand:
        if len(neg) >= N_SPOT_NEG:
            break
        H = m["hom"][fr]
        for sx, sy in SPOTS:
            x, y = sx / PITCH[0], sy / PITCH[1]
            dd = H[2, 0] * x + H[2, 1] * y + H[2, 2]
            su, sv = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / dd, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / dd
            bx, by = m["bx"][fr] * PITCH[0], m["by"][fr] * PITCH[1]
            if 20 < su < 1900 and 20 < sv < 1060 and np.hypot(bx - sx, by - sy) > 15:
                neg.append((fr, su, sv, "spot"))
                break
    for fr in rng.choice(gframes, N_RANDOM_NEG, replace=False):
        neg.append((fr, rng.uniform(0, 1920), rng.uniform(0, 1080), "random"))
    jobs = {}
    for r in pos.itertuples():
        jobs.setdefault(int(r.frame), []).append(("pos", r.x, r.y, r.w, r.h))
    for fr, su, sv, kind in neg:
        jobs.setdefault(int(fr), []).append((kind, su, sv, 0, 0))
    by_frame = {fr: g for fr, g in balls[(balls.conf >= 0.3) & ~balls.spot].groupby("frame")}
    for s in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", s), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", s), exist_ok=True)
    cap = cv2.VideoCapture(asset_paths()["video"])
    counts = {}
    last = max(jobs)
    for fr in range(last + 1):
        if not cap.grab():
            break
        if fr not in jobs:
            continue
        img = cap.retrieve()[1]
        s = split_of(fr)
        for j, (kind, x, y, w, h) in enumerate(jobs[fr]):
            x0, y0 = tile_origin(rng, x, y)
            lines = []
            if kind == "pos":
                lines.append("0 {:.6f} {:.6f} {:.6f} {:.6f}".format((x - x0) / TILE, (y - y0) / TILE, w / TILE, h / TILE))
            else:
                if x0 <= u[fr] < x0 + TILE and y0 <= v[fr] < y0 + TILE:
                    continue                    # the ATD ball is inside
                others = by_frame.get(fr)
                if others is not None and np.any((others.x > x0) & (others.x < x0 + TILE) & (others.y > y0) & (others.y < y0 + TILE)):
                    continue                    # a possible ball inside: not a clean negative
            name = "{}_{:06d}_{}".format(kind, fr, j)
            cv2.imwrite(os.path.join(OUT, "images", s, name + ".jpg"), img[y0:y0 + TILE, x0:x0 + TILE], [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(OUT, "labels", s, name + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            counts[(s, kind)] = counts.get((s, kind), 0) + 1
    with open(os.path.join(OUT, "data.yaml"), "w") as fh:
        fh.write("path: {}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n".format(OUT))
    print("tiles:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
