"""P9co: gate C of the video evidence on our match (notes/BULGULAR.md, 9 Nov), without any label: the dense video
evidence of P9cm (cache/barath_dense.npz, a model trained on SoccerTrack only) against v19c's own chain
(cache/v19c_dump.npz: p with the restart evidence, switch costs, dead ball).
  1. where the chain is sure (p > 0.9 or < 0.1, live frames), the evidence's team agrees on >= 80 % of the frames;
  2. the fused chain's possession count is within 0.8-1.5 x v19c's.
Also: the share of frames the fusion changes and the count of possessions shorter than 2 s. The public labels are not
read. With an output path, writes the fused chain as a submission CSV (only meaningful after gates A / B hold).

    python analysis/p9co_barath_gateC.py [w] [<out csv>]
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah.teammodel import decode  # noqa: E402

FPS = 25
NAMES = {pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}


def spells(c):
    e = np.flatnonzero(np.diff(c) != 0) + 1
    a, b = np.r_[0, e], np.r_[e, len(c)]
    k = np.isin(c[a], (pm.H, pm.A))
    return a[k], b[k]


def main(w=0.5, out=None):
    d = np.load(os.path.join(ROOT, "cache", "v19c_dump.npz"))
    p, sw, dead, g = d["p"].astype(np.float64), d["switch"], d["dead"], d["g"]
    e = np.load(os.path.join(ROOT, "cache", "barath_dense.npz"))["e"].astype(np.float64)
    dead0 = dead
    if os.environ.get("DEAD"):                                   # the fused chain with another dead ball (P9cu); v19c keeps its own
        dead = np.load(os.environ["DEAD"])["dead"]
    alive = g & ~dead
    lo = np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))
    base = np.where(g, decode(p, g & ~dead0, sw), pm.X).astype(np.int8)
    new = np.where(g, decode(1 / (1 + np.exp(-(lo + w * e))), alive, sw), pm.X).astype(np.int8)
    sure = alive & ((p > 0.9) | (p < 0.1)) & (e != 0)
    agree = float(np.mean((e[sure] > 0) == (p[sure] > 0.5)))
    (a0, b0), (a1, b1) = spells(base), spells(new)
    ratio = len(a1) / len(a0)
    print("evidence on {:.1%} of the live frames; |e| median {:.2f}".format(np.mean(e[alive] != 0), np.median(np.abs(e[alive]))))
    print("1. agreement with the chain where it is sure ({:.1%} of the live frames): {:.1%} -> {}".format(sure.sum() / alive.sum(), agree, "holds" if agree >= 0.8 else "FAILS"))
    print("2. possessions v19c {} fused {} (x {:.2f}) -> {}".format(len(a0), len(a1), ratio, "holds" if 0.8 <= ratio <= 1.5 else "FAILS"))
    print("frames changed {:.2%} of the live frames; possessions shorter than 2 s: v19c {} fused {}".format(
        np.mean(base[g] != new[g]) * g.sum() / alive.sum(), int(np.sum(b0 - a0 < 2 * FPS)), int(np.sum(b1 - a1 < 2 * FPS))))
    if out:
        pd.DataFrame({"frame": np.arange(len(new)), "label": [NAMES[int(c)] for c in new]}).to_csv(out, index=False)
        print("->", out)
    return 0 if agree >= 0.8 and 0.8 <= ratio <= 1.5 else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(float(a[0]) if a else 0.5, a[1] if len(a) > 1 else None))
