"""Step 2a: self-labels for a "this person has the ball" crop classifier (Peral et al. 2025/2026). GT-free.

    uv run python analysis/p6c_possessor_labels.py      # -> cache/possessor_labels.npz

No person decides any label (RULES.md: pseudo-labels produced by an algorithm). On frames where the
ATD sees the ball on the pitch, slower than SLOW (a ball at someone's feet or loose, not a pass in
flight), each pretrained-YOLO person box (cache/yolo) is labelled from its foot point projected to
the pitch: positive when it is the nearest person to the ATD ball, within POS_M, and the next person
is CLEAR_M or more away; negative when it is NEG_M or more from the ball. Everything else is left
out. Split as ball_v1: every 5th 10-minute block is validation; the public window (+-10 s) is out.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from mtah.data import PITCH, load_tactical  # noqa: E402
from p3a_yolo_ball import to_pitch  # noqa: E402
from p3e_ball_pseudolabels import BLOCK, PUBLIC  # noqa: E402

YOLO = os.path.join(ROOT, "cache", "yolo")
OUT = os.path.join(ROOT, "cache", "possessor_labels.npz")
SLOW = 6.0              # m/s: ball at feet or loose; a pass in flight is faster
POS_M, CLEAR_M, NEG_M = 1.0, 2.0, 3.0
MIN_CONF = 0.5
SEED = 0


def main():
    m = load_tactical()
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(YOLO, "chunk_*.npz")))]
    b = pd.DataFrame({c: np.concatenate([p[c] for p in parts]) for c in ("frame", "cls", "conf", "x", "y", "w", "h")})
    b = b[(b.cls == 0) & (b.conf >= MIN_CONF)].drop(columns="cls").reset_index(drop=True)
    foot = pd.DataFrame({"frame": b.frame.values, "x": b.x.values, "y": b.y.values + b.h.values / 2})
    b["X"], b["Y"] = to_pitch(foot, m["hom"])
    bx, by = m["bx"] * PITCH[0], m["by"] * PITCH[1]
    # the ATD's own speed field is mostly empty: speed from positions, over 5 frames
    sp = np.full(len(bx), np.nan)
    sp[5:] = np.hypot(bx[5:] - bx[:-5], by[5:] - by[:-5]) * 25 / 5
    f = b.frame.values
    ok = ~np.isnan(bx[f]) & (np.nan_to_num(sp[f], nan=99) < SLOW) & m["graded"][f]
    ok &= (bx[f] >= 0) & (bx[f] <= PITCH[0]) & (by[f] >= 0) & (by[f] <= PITCH[1])
    ok &= (f < PUBLIC[0] - 250) | (f > PUBLIC[1] + 250)
    b = b[ok].reset_index(drop=True)
    b["d"] = np.hypot(b.X.values - bx[b.frame.values], b.Y.values - by[b.frame.values])
    b["rank"] = b.groupby("frame").d.rank(method="first")
    second = b[b["rank"] == 2].set_index("frame").d
    b["second"] = second.reindex(b.frame.values).values
    pos = (b["rank"] == 1) & (b.d <= POS_M) & (np.nan_to_num(b.second.values, nan=np.inf) >= CLEAR_M)
    neg = b.d >= NEG_M
    b["label"] = np.where(pos, 1, np.where(neg, 0, -1))
    b = b[b.label >= 0].reset_index(drop=True)
    b["val"] = (b.frame.values // BLOCK) % 5 == 4
    # every positive; negatives 3 per positive per split, half of them hard (3-8 m from the ball)
    rng = np.random.default_rng(SEED)
    keep = []
    for v in (False, True):
        s = b[b.val == v]
        p = s[s.label == 1]
        hard, easy = s[(s.label == 0) & (s.d < 8)], s[(s.label == 0) & (s.d >= 8)]
        k = int(1.5 * len(p))
        keep += [p, hard.iloc[rng.permutation(len(hard))[:k]], easy.iloc[rng.permutation(len(easy))[:k]]]
    out = pd.concat(keep).sort_values(["frame", "x"]).reset_index(drop=True)
    np.savez_compressed(OUT, **{c: out[c].values for c in ("frame", "x", "y", "w", "h", "d", "label", "val")})
    print(out.groupby(["val", "label"]).size().to_string())
    print("frames", out.frame.nunique(), "->", OUT)


if __name__ == "__main__":
    main()
