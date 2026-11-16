"""Error kinds of the v9 recipe in the full chain (DFL, carried dead ball), from P7l's cached predictions.

    python analysis/p7r_error_kinds_v9.py      # -> eda_out/reports/P7r_error_kinds_v9.md

Every true possession, by duration: found at IoU 0.5; mostly called dead (> half its frames D); given to the
wrong team (no same-team overlap); merged with a neighbour; split; shifted. Also with the dead ball from
the labels, to separate the two halves of the chain.
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
from mtah import grumodel as G  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p6w_error_kinds import BINS, IOU, bin_of  # noqa: E402
from p7l_dead_blocks import blocks, preds  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P7r_error_kinds_v9.md")
KINDS = ("bulundu", "ölü denmiş", "yanlış takım", "komşuyla birleşmiş", "bölünmüş", "kaymış")


def kinds(labels, c):
    runs = pm.valid_runs(labels)
    gs, ps = pm.possession_spells(labels, runs), pm.possession_spells(c, runs)
    ov = pm.spell_overlaps(gs, ps)
    flags = pm.matched_gt_flags(ov, len(gs), IOU)
    best = {}
    for iou, i, j in ov:
        best.setdefault(i, j)
    out = []
    for i, (a, b, k) in enumerate(gs):
        if flags[i]:
            kind = "bulundu"
        elif np.mean(c[a:b] == pm.D) > 0.5:
            kind = "ölü denmiş"
        elif i not in best:
            kind = "yanlış takım"
        else:
            pa, pb, _ = ps[best[i]]
            r = (pb - pa) / max(1, b - a)
            kind = "komşuyla birleşmiş" if r > 2 else "bölünmüş" if r < 0.5 else "kaymış"
        out.append({"bin": bin_of((b - a) / 25), "kind": kind})
    return out


def main():
    P = preds()
    rows = {"tam zincir (v9)": [], "ölü top etiketten": []}
    for t, d in P.items():
        g = d["g"]
        cost = G.SWITCH_HI * (1 - d["b"]) + G.SWITCH_LO * d["b"]
        dead = blocks((d["qh"] + d["qg"]) / 2, 0.5, 2, 2) & g
        for name, dd in (("tam zincir (v9)", dead), ("ölü top etiketten", d["labels"] == pm.D)):
            c = np.where(g, decode(d["p"], g & ~dd, cost), pm.X).astype(np.int8)
            rows[name] += kinds(d["labels"], c)
    names = ["{}–{} s".format(BINS[i], int(BINS[i + 1]) if BINS[i + 1] < 1e8 else "∞") for i in range(len(BINS) - 1)]
    lines = ["# P7r: v9 tarifinin hata türleri (taşınan ölü toplu DFL, 7 maç, IoU 0.5)", "", "`analysis/p7r_error_kinds_v9.py`."]
    for name, r in rows.items():
        tab = pd.DataFrame(r).groupby(["bin", "kind"]).size().unstack(fill_value=0)
        ks = [k for k in KINDS if k in tab.columns]
        lines += ["", "## " + name, "", "| süre | sayı | " + " | ".join(ks) + " |", "|---|---|" + "---|" * len(ks)]
        for b, rr in tab.iterrows():
            n = int(rr.sum())
            lines.append("| {} | {} | ".format(names[b], n) + " | ".join("{} ({:.0%})".format(int(rr.get(k, 0)), rr.get(k, 0) / n) for k in ks) + " |")
        tot = tab.sum()
        lines.append("| hepsi | {} | ".format(int(tot.sum())) + " | ".join("{} ({:.0%})".format(int(tot.get(k, 0)), tot.get(k, 0) / tot.sum()) for k in ks) + " |")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
