"""Inventory: "the degradation is realistic" rested on one number against the public window (baseline
0.319 on degraded Metrica vs 0.326 public). A GT-free check in the labelers' output instead.

    uv run python analysis/p5t_degrade_realism.py      # -> eda_out/reports/P5t_degrade_realism.md

The same labelers (organiser's baseline_smooth50, nearest player, control rule) on our match and on
degraded external data, all with their own dead ball so nothing comes from labels: possessions per
90 graded minutes, median possession length, share under 2 s, and how often the label flips. If the
degradation makes external data look like our ATD, these output statistics should look alike too.
Three seeds; dead ball for the external data from the labels' D would leak, so own_dead is used
everywhere (tracking-only rule), and on our match the same.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.labelers import baseline_smooth, control, nearest  # noqa: E402
from mtah.match import from_tactical  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "eda_out", "reports", "P5t_degrade_realism.md")


def stats(m, codes):
    g = m["half"] > 0
    c = np.where(g, codes, pm.X).astype(np.int8)
    runs = pm.valid_runs(np.where(g, pm.D, pm.X).astype(np.int8))
    sp = pm.possession_spells(c, runs)
    lens = np.array([(b - a) / FPS for a, b, _ in sp])
    minutes = g.sum() / FPS / 60
    flips = np.sum(c[1:][g[1:] & g[:-1]] != c[:-1][g[1:] & g[:-1]])
    return {"per90": len(sp) / minutes * 90, "median_s": np.median(lens) if len(lens) else np.nan,
            "short": np.mean(lens < 2) if len(lens) else np.nan, "flips_per_min": flips / minutes}


def main():
    labelers = {"baseline_smooth50": lambda m: baseline_smooth(m, 50), "en yakın": nearest, "kontrol (kendi ölü topu)": lambda m: control(m)}
    rows = []
    ours = from_tactical()
    for name, f in labelers.items():
        rows.append(dict(stats(ours, f(ours)), src="bizim maç", labeler=name, seed=0))
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ext = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2)]
    for seed in (0, 1, 2):
        for cm, key in ext:
            d = degrade(cm, rates_for(key), seed=seed)
            for name, f in labelers.items():
                rows.append(dict(stats(d, f(d)), src="Metrica bozulmuş" if key == "metrica" else "DFL bozulmuş", labeler=name, seed=seed))
        print("seed", seed, flush=True)
    t = pd.DataFrame(rows).groupby(["labeler", "src"])[["per90", "median_s", "short", "flips_per_min"]].mean()
    lines = ["# P5t: bozma modeli etiketleyici çıktısında gerçekçi mi? (GT'siz)", "",
             "`analysis/p5t_degrade_realism.py`. Hepsi kendi ölü topuyla (`own_dead`, sadece tracking); dış veri 3 tohum ortalaması.", "",
             "| etiketleyici | kaynak | 90 dk'da possession | medyan süre s | < 2 s payı | dakikada etiket değişimi |", "|---|---|---|---|---|---|"]
    for (lab, src), r in t.iterrows():
        lines.append("| {} | {} | {:.0f} | {:.1f} | {:.0%} | {:.1f} |".format(lab, src, r["per90"], r["median_s"], r["short"], r["flips_per_min"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
