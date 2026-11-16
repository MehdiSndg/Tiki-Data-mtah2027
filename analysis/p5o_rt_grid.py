"""Inventory: the control rule's R and T disagreed between sources (DFL could not tell 1.0/0.2 from
1.5/0.4, Metrica preferred 1.0/0.2). Again with three sources, three seeds, and the ATD-like
smoothed ball.

    uv run python analysis/p5o_rt_grid.py      # -> eda_out/reports/P5o_rt_grid.md

Team call only (dead ball from the labels), degraded as in P2, and degraded with the ball smoothed
over 25 frames (our ATD's ball is about that smooth). Sources: DFL (7), Metrica games 1-2, Metrica
game 3 (EPTS, a third match). spellF1, mean over matches, then mean [min-max] over seeds.
"""

import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.labelers import control  # noqa: E402

RS, TS, SEEDS = (1.0, 1.5, 2.0), (0.2, 0.4, 0.6), (0, 1, 2)
OUT = os.path.join(ROOT, "eda_out", "reports", "P5o_rt_grid.md")


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    src = {"DFL": [(from_dfl(i), i) for i in ids], "Metrica 1–2": [(from_metrica_sample(g), "metrica") for g in (1, 2)],
           "Metrica 3": [(from_metrica_sample(3), "metrica")]}
    rows = []
    for seed in SEEDS:
        for smooth in (0, 25):
            for s, ms in src.items():
                for m, key in ms:
                    r = dict(rates_for(key), smooth_frames=smooth)
                    d = degrade(m, r, seed=seed)
                    for R in RS:
                        for T in TS:
                            p = control(d, R=R, T=T, dead="truth")
                            rows.append({"seed": seed, "smooth": smooth, "src": s, "R": R, "T": T,
                                         "f1": pm.spell_scores(d["labels"], p, pm.valid_runs(d["labels"]))["mean_f1"]})
            print(seed, flush=True)
    df = pd.DataFrame(rows).groupby(["smooth", "src", "R", "T", "seed"]).f1.mean().reset_index()
    t = df.groupby(["smooth", "src", "R", "T"]).f1.agg(["mean", "min", "max"]).reset_index()
    lines = ["# P5o: kontrol kuralının R ve T'si, üç kaynak, üç tohum", "",
             "`analysis/p5o_rt_grid.py`. Takım kararı (ölü top etiketten), bozulmuş veri; spellF1 ortalama [tohumlar arası "
             "en düşük–en yüksek]. **Kalın:** kaynağın en iyisi.", ""]
    for smooth in (0, 25):
        lines += ["## Top yumuşatma {} kare".format(smooth), "", "| kaynak | " + " | ".join("R {} / T {}".format(R, T) for R in RS for T in TS) + " |",
                  "|---|" + "---|" * (len(RS) * len(TS))]
        for s in src:
            sub = t[(t.smooth == smooth) & (t.src == s)]
            best = sub["mean"].max()
            cells = []
            for R in RS:
                for T in TS:
                    r = sub[(sub.R == R) & (sub["T"] == T)].iloc[0]
                    c = "{:.3f} [{:.3f}–{:.3f}]".format(r["mean"], r["min"], r["max"])
                    cells.append("**{}**".format(c) if r["mean"] == best else c)
            lines.append("| {} | {} |".format(s, " | ".join(cells)))
        lines.append("")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
