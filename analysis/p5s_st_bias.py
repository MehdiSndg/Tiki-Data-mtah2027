"""Is the Smart Tagging side a judge that rewards fewer hand-overs rather than correct ones?

    uv run python analysis/p5s_st_bias.py      # -> eda_out/reports/P5s_st_bias.md

The control rule's T (seconds of control before a hand-over) moves the number of possessions
monotonically. On our match (GT-free): possessions per 90 minutes and Smart Tagging agreement for
each T, with the raw-ATD classifier's dead ball. On degraded external data, ATD-like smoothed ball
(25 frames), dead ball from the labels: the real spellF1 for the same T, 3 seeds. A judge that keeps
rising with T while spellF1 peaks and falls rewards stickiness.
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
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402
from p4_common import ours  # noqa: E402

TS = (0.2, 0.4, 0.6, 0.8, 1.2, 2.0, 3.0)
OUT = os.path.join(ROOT, "eda_out", "reports", "P5s_st_bias.md")


def main():
    m = from_tactical()
    df = load_frames()
    dead, _ = ours()
    played = (m["half"] > 0) & ~np.isin(df["period"].values, NOT_PLAYED)
    alive = played & ~dead
    tag = np.where(df["tag_H"].values > df["tag_A"].values, pm.H, np.where(df["tag_A"].values > df["tag_H"].values, pm.A, -1))
    outside = np.ones(len(dead), bool)
    outside[11185:18685] = False
    judged = alive & outside & (tag >= 0)
    minutes = played.sum() / 25 / 60
    ours_rows = []
    for T in TS:
        c = np.where(m["half"] > 0, control(m, T=T, dead=~alive), pm.X).astype(np.int8)
        spells = pm.possession_spells(c, pm.valid_runs(np.where(m["half"] > 0, pm.D, pm.X).astype(np.int8)))
        ours_rows.append((T, len(spells) / minutes * 90, np.mean(c[judged] == tag[judged])))
        print(ours_rows[-1], flush=True)
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ext = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2)]
    rows = []
    for seed in (0, 1, 2):
        for cm, key in ext:
            d = degrade(cm, dict(rates_for(key), smooth_frames=25), seed=seed)
            for T in TS:
                p = control(d, T=T, dead="truth")
                rows.append({"src": "Metrica" if key == "metrica" else "DFL", "seed": seed, "T": T,
                             "f1": pm.spell_scores(d["labels"], p, pm.valid_runs(d["labels"]))["mean_f1"]})
    e = pd.DataFrame(rows).groupby(["src", "T", "seed"]).f1.mean().groupby(["src", "T"]).mean().unstack(0)
    lines = ["# P5s: Smart Tagging az değişimi mi ödüllendiriyor?", "",
             "`analysis/p5s_st_bias.py`. Kontrol kuralı R 1.5 m, T değişiyor. Bizim maç: GT'siz. Dış veri: bozulmuş + 25 kare "
             "yumuşatılmış top, ölü top etiketten, 3 tohum ortalaması.", "",
             "| T s | bizim maç: 90 dk'da possession | bizim maç: Smart Tagging uyumu | DFL spellF1 | Metrica spellF1 |", "|---|---|---|---|---|"]
    for (T, n, st) in ours_rows:
        lines.append("| {} | {:.0f} | {:.1%} | {:.3f} | {:.3f} |".format(T, n, st, e.loc[T, "DFL"], e.loc[T, "Metrica"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
