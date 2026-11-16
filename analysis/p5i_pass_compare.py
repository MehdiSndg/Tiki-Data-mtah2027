"""Two full-match detector passes, frame by frame: how far apart are their ball boxes?

    uv run python analysis/p5i_pass_compare.py cache/ball_v1_pass <other pass dir> [--conf 0.3]

Made for the strict-fp32 Blackwell pass (Colab, 25 Oct) against the local RTX 4060 pass of 23 Oct
(TF32 on; notes/BULGULAR.md). Per graded frame, the most confident ball box (class 0) at or
above --conf in each pass: frames with a box in one pass only, and for frames with both, the
centre distance and the confidence difference. Both manifests are printed when present.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def best_boxes(d, conf):
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(d, "chunk_*.npz")))]
    b = pd.DataFrame({c: np.concatenate([p[c] for p in parts]) for c in ("frame", "cls", "conf", "x", "y")})
    b = b[(b.cls == 0) & (b.conf >= conf)]
    return b.sort_values(["frame", "conf"], kind="stable").groupby("frame").tail(1).set_index("frame"), len(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()
    for d in (args.a, args.b):
        m = os.path.join(d, "manifest.json")
        print(d, json.load(open(m)) if os.path.exists(m) else "(no manifest)")
    A, na = best_boxes(args.a, args.conf)
    B, nb = best_boxes(args.b, args.conf)
    both = A.index.intersection(B.index)
    d = np.hypot(A.loc[both, "x"].values - B.loc[both, "x"].values, A.loc[both, "y"].values - B.loc[both, "y"].values)
    dc = np.abs(A.loc[both, "conf"].values - B.loc[both, "conf"].values)
    print("| | değer |\n|---|---|")
    print("| parça sayısı A / B | {} / {} |".format(na, nb))
    print("| top kutusu olan kare A / B | {} / {} |".format(len(A), len(B)))
    print("| sadece A'da / sadece B'de | {} / {} |".format(len(A.index.difference(B.index)), len(B.index.difference(A.index))))
    print("| ikisinde de | {} |".format(len(both)))
    for t in (0.01, 0.5, 2, 10):
        print("| merkez farkı > {} px | {} ({:.3%}) |".format(t, int((d > t).sum()), np.mean(d > t)))
    print("| en büyük merkez farkı px (aynı top) | {:.3f} |".format(np.max(d[d <= 10]) if np.any(d <= 10) else 0))
    print("| güven farkı medyan / p99 / en büyük | {:.5f} / {:.5f} / {:.5f} |".format(np.median(dc), np.percentile(dc, 99), dc.max()))


if __name__ == "__main__":
    main()
