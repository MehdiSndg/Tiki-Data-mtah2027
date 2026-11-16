"""A placed-ball rule for the dead ball: the ATD ball still at a restart place. GT-free; thresholds
from the literature, fixed in advance.

    uv run python analysis/p6p_dead_placed_ball.py      # -> eda_out/reports/P6p_dead_placed_ball.md

Dead where the ATD ball is seen, slower than STILL (0.5 m/s, mtah.degrade's still ball) over a centred
2 s window, and at a restart place (Vidal-Codina et al. 2022, S4 tolerances): within 0.5 m outside to
2 m inside a touchline (throw-in), 3 m of a corner, inside a goal area (goal kick), 2 m of a penalty
spot, 4 m of the centre spot (our kickoff ball tolerance). Blocks under 2 s dropped. Added to the v3
dead ball (v2 + broadcast rule). Judged as P6o: Smart Tagging set pieces with a dead ball in the 5 s
before them, the dead share, where the added frames sit; the public window afterwards as a check.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.data import PITCH, load_tactical  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from p4_common import dead_v3  # noqa: E402
from p4d_still_and_kinks import runs_of  # noqa: E402

STILL, MIN_BLOCK = 0.5, 50
OUT = os.path.join(ROOT, "eda_out", "reports", "P6p_dead_placed_ball.md")


def placed_ball_dead(m, half):
    L, W = PITCH
    x, y = m["bx"].astype(float) * L, m["by"].astype(float) * W
    sp = np.full(len(x), np.nan)
    sp[1:] = np.hypot(np.diff(x), np.diff(y)) * 25
    still = pd.Series(sp).rolling(50, center=True, min_periods=40).max().values < STILL
    touch = (y < 2) | (y > W - 2)
    corner = np.min([np.hypot(x - cx, y - cy) for cx in (0, L) for cy in (0, W)], axis=0) < 3
    goal_area = ((x < 5.5) | (x > L - 5.5)) & (np.abs(y - W / 2) < 9.16)
    pen = np.min([np.hypot(x - px, y - W / 2) for px in (11, L - 11)], axis=0) < 2
    centre = np.hypot(x - L / 2, y - W / 2) < 4
    rule = still & (touch | corner | goal_area | pen | centre) & (half > 0)
    for a, z in runs_of(rule):
        if z - a < MIN_BLOCK:
            rule[a:z] = False
    return rule


def main():
    fr = load_frames()
    m = load_tactical()
    g = fr["half"].values > 0
    dead = dead_v3()
    rule = placed_ball_dead(m, fr["half"].values)
    new = rule & ~dead
    out = np.ones(len(g), bool)
    out[11185:18685] = False
    starts = [a for a, _ in runs_of(fr["setpiece"].values.astype(bool)) if out[a] and g[a]]
    cov = lambda d: np.mean([d[max(0, a - 125):a + 1].any() for a in starts])
    lines = ["# P6p: yerleştirilmiş top kuralı (GT'siz, eşikler literatürden)", "", "`analysis/p6p_dead_placed_ball.py`.", "",
             "| | ölü kare payı (public dışı) | duran topların önünde ölü top |", "|---|---|---|",
             "| v3 ölü top | {:.1%} | {:.1%} ({}) |".format(dead[g & out].mean(), cov(dead), len(starts)),
             "| v3 + kural | {:.1%} | {:.1%} |".format((dead | rule)[g & out].mean(), cov(dead | rule)), "",
             "Kuralın eklediği {} kare (public dışı {}), ortalama blok {:.1f} s.".format(
                 new.sum(), (new & out).sum(), np.mean([(z - a) / 25 for a, z in runs_of(new)]) if new.any() else 0)]
    gt = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    w = slice(11185, 18685)
    t = gt[w] == pm.D
    for name, d in (("v3", dead), ("v3 + kural", dead | rule)):
        o = d[w]
        lines.append("Public (kontrol): {} — GT ölüyü yakalama {:.0%}, ölü dediğimizin GT ölü payı {:.0%}.".format(name, np.mean(o[t]), np.mean(t[o]) if o.any() else 0))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
