"""A broadcast + stillness rule for the dead ball the classifier misses while the ATD keeps tracking a
dead ball. On our match, GT-free; thresholds fixed in advance.

    uv run python analysis/p6o_dead_rule.py      # -> eda_out/reports/P6o_dead_rule.md

Rule: dead where, over a centred 3 s window, the broadcast shows no pitch on >= AWAY of the frames it
has, and the median player speed over 2 s is below SLOW (1.3 m/s: the DFL-derived stillness threshold
of analysis/v0_pipeline.py); blocks shorter than 2 s dropped, as the classifier's. Added to the dead-ball
v2 mask. Judged GT-free on the frames outside the public window: the share of Smart Tagging set pieces
with a dead ball in the 5 s before them, the dead share, and where the added frames sit (ATD ball
seen or not, ball speed). The public window is printed afterwards as a correctness check only.
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
from mtah.frames import load_frames  # noqa: E402
from mtah.deadball import broadcast_still_dead  # noqa: E402
from p4_common import dead_v2  # noqa: E402
from p4d_still_and_kinks import runs_of  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P6o_dead_rule.md")


def main():
    fr = load_frames()
    g = fr["half"].values > 0
    dead = dead_v2()
    rule = broadcast_still_dead(fr, os.path.join(ROOT, "cache", "broadcast_on_tactical.npz"))
    new = rule & ~dead
    out = np.ones(len(g), bool)
    out[11185:18685] = False
    starts = [a for a, _ in runs_of(fr["setpiece"].values.astype(bool)) if out[a] and g[a]]
    cov = lambda d: np.mean([d[max(0, a - 125):a + 1].any() for a in starts])
    seen = fr["ball_status"].values == 0
    lines = ["# P6o: yayın + duran oyuncular kuralı (GT'siz, eşikler önceden)", "", "`analysis/p6o_dead_rule.py`.", "",
             "| | ölü kare payı (public dışı) | duran topların önünde ölü top |", "|---|---|---|",
             "| ölü top v2 | {:.1%} | {:.1%} ({}) |".format(dead[g & out].mean(), cov(dead), len(starts)),
             "| v2 + kural | {:.1%} | {:.1%} |".format((dead | rule)[g & out].mean(), cov(dead | rule)), "",
             "Kuralın eklediği {} kare (public dışı {}): ATD topu görüyor {:.0%}, top hızı medyan {:.1f} m/s, ortalama blok {:.1f} s.".format(
                 new.sum(), (new & out).sum(), seen[new].mean(), np.nanmedian(fr["ball_speed"].values[new]),
                 np.mean([(z - a) / 25 for a, z in runs_of(new)]) if new.any() else 0)]
    gt = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    w = slice(11185, 18685)
    t = gt[w] == pm.D
    for name, d in (("v2", dead), ("v2 + kural", dead | rule)):
        o = d[w]
        lines.append("Public (kontrol): {} — GT ölüyü yakalama {:.0%}, ölü dediğimizin GT ölü payı {:.0%}.".format(name, np.mean(o[t]), np.mean(t[o]) if o.any() else 0))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
