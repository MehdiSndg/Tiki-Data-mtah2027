"""P9cn: are v19's short possessions next to a dead ball worth keeping? (external data only; asked after two such
predictions missed in the public window, which is not used.) v19's out-of-fold chain on DFL and Metrica; every predicted
possession shorter than T seconds that touches a dead block (starts right after one or ends right before one) is
relabelled dead; spellF1 at 1.6 s and 3 s with and without. Also their precision: the share matched at IoU 0.5.

    python analysis/p9cn_dead_adjacent_short.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p9b_oof_sweeps import HI, LO  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

C = lambda *p: os.path.join(ROOT, "cache", *p)
FPS = 25


def runs_of(c):
    e = np.flatnonzero(np.diff(c) != 0) + 1
    return list(zip(np.r_[0, e], np.r_[e, len(c)]))


def main():
    res = {}
    for f in sorted(glob.glob(C("oof_tcn128d3", "*.npz"))):
        n = os.path.basename(f)
        z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f))
        g, lab = z["g"], z["labels"]
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        l3 = merged(lab, 3.0)
        p = (big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6
        c = np.where(g, decode(p, g & ~dead, HI * (1 - z["b"]) + LO * z["b"]), pm.X).astype(np.int8)
        sc = lambda x: np.array([pm.spell_scores(lab, x, pm.valid_runs(lab))["mean_f1"], pm.spell_scores(l3, x, pm.valid_runs(l3))["mean_f1"]])
        base = sc(c)
        R = runs_of(c)
        src = "DFL" if "DFL" in n else "Metrica"
        for T in (1.0, 2.0, 3.0):
            x = c.copy()
            k = hit = 0
            for i, (a, b) in enumerate(R):
                if c[a] not in (pm.H, pm.A) or b - a >= T * FPS:
                    continue
                if (i > 0 and c[R[i - 1][0]] == pm.D) or (i + 1 < len(R) and c[R[i + 1][0]] == pm.D):
                    x[a:b] = pm.D
                    k += 1
                    seg = lab[a:b]
                    # precision proxy: a label possession of the same team overlapping with IoU >= 0.5
                    aa, bb = a, b
                    while aa > 0 and lab[aa - 1] == c[a]:
                        aa -= 1
                    while bb < len(lab) and lab[bb] == c[a]:
                        bb += 1
                    inter = int(np.sum(seg == c[a]))
                    hit += inter > 0 and inter / max((bb - aa) + (b - a) - inter, 1) >= 0.5
            res.setdefault((src, T), []).append((*(sc(x) - base), k, hit))
        print(n, "base {:.4f} {:.4f}".format(*base), flush=True)
    print("data, shorter than | removing them: gain 1.6 s, 3 s (matches ahead) | count per match | matched at IoU 0.5")
    for (src, T), v in res.items():
        v = np.array(v, float)
        print("{} < {} s | {:+.4f} {:+.4f} ({} / {}) | {:.1f} | {:.0%}".format(src, T, v[:, 0].mean(), v[:, 1].mean(), int((v[:, 0] > 0).sum()), len(v),
                                                                       v[:, 2].mean(), v[:, 3].sum() / max(v[:, 2].sum(), 1)))


if __name__ == "__main__":
    main()
