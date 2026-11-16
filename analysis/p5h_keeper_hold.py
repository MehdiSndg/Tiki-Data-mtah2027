"""Inventory E29: when a keeper holds the ball, the ATD drops it -- does the dead-ball classifier then call play dead?

    uv run python analysis/p5h_keeper_hold.py      # -> eda_out/reports/P5h_keeper_hold.md

On our match this rested on 9 gaps checked by eye (6 in play, v0 called 5 of them dead). Here on
degraded external data with real labels: frames where the true ball is within GK_M of a keeper
for at least GK_S in play and the degradation dropped it, against the other in-play frames without
a ball and the in-play frames with one. The m1 classifier is trained cross-dataset (as P4a, P5g);
three degradation seeds.
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
from m1_dead_classifier import table  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from p4a_error_budget import classifier_dead  # noqa: E402

FPS = 25
GK_M, GK_S = 1.0, 0.5
SEEDS = (0, 1, 2)
OUT = os.path.join(ROOT, "eda_out", "reports", "P5h_keeper_hold.md")


def held(m):
    L, W = m["pitch"]
    near = np.zeros(len(m["bx"]), bool)
    for k in m["gk"].values():
        if k >= 0:
            near |= np.hypot((m["px"][:, k] - m["bx"]) * L, (m["py"][:, k] - m["by"]) * W) <= GK_M
    out = np.zeros_like(near)
    e = np.diff(np.r_[0, near.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if b - a >= GK_S * FPS:
            out[a:b] = True
    return out


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    clean = {"DFL": [(from_dfl(i), i) for i in ids], "Metrica": [(from_metrica_sample(g), "metrica") for g in (1, 2)]}
    rows = []
    for seed in SEEDS:
        deg = {s: [degrade(m, rates_for(k), seed=seed) for m, k in ms] for s, ms in clean.items()}
        clf = {s: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed).fit(*table(v))
               for s, v in deg.items()}
        for s, ms in clean.items():
            other = clf["Metrica" if s == "DFL" else "DFL"]
            for (m, _), d in zip(ms, deg[s]):
                alive = np.isin(m["labels"], (pm.H, pm.A))
                dead = classifier_dead(other, d)
                gone = np.isnan(d["bx"])
                hold = alive & held(m) & gone
                rows.append({"src": s, "seed": seed, "n_hold": hold.sum() / alive.sum(),
                             "hold": dead[hold].mean() if hold.any() else np.nan,
                             "other_gap": dead[alive & gone & ~hold].mean(), "ball": dead[alive & ~gone].mean(),
                             "true_dead": dead[m["labels"] == pm.D].mean()})
            print(seed, s, flush=True)
    df = pd.DataFrame(rows)
    t = df.groupby("src")[["n_hold", "hold", "other_gap", "ball", "true_dead"]].agg(["mean", "min", "max"])
    lines = ["# P5h: kaleci topu tutarken ölü top sınıflandırıcısı ne diyor? (envanter E29)", "",
             "`analysis/p5h_keeper_hold.py`. Bozulmuş dış veri, gerçek etiket; sınıflandırıcı çapraz veri setiyle eğitildi. "
             "Kaleci tutuşu: gerçek top kaleciye ≤ {:.0f} m, ≥ {} s, oyunda, ve bozma topu silmiş. "
             "{} tohum; ortalama, köşeli parantezde maçlar ve tohumlar arasındaki en düşük–en yüksek.".format(GK_M, GK_S, len(SEEDS)), "",
             "| kaynak | tutuşun oyun içi payı | sınıflandırıcının ölü dediği pay: kaleci tutuyor | oyunda, başka boşluk | oyunda, top var | gerçek ölü top |",
             "|---|---|---|---|---|---|"]
    for s, r in t.iterrows():
        cell = lambda c, f="{:.0%}": "{} [{}–{}]".format(f.format(r[(c, "mean")]), f.format(r[(c, "min")]), f.format(r[(c, "max")]))
        lines.append("| {} | {} | {} | {} | {} | {} |".format(s, cell("n_hold", "{:.2%}"), cell("hold"), cell("other_gap"),
                                                         cell("ball"), cell("true_dead")))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
