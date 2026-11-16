"""Explicit-duration decoding (mtah.hsmm) against v9's Viterbi, on P7l's cached v9 predictions (full chain,
carried-dead-ball DFL). CPU.

    python analysis/p7w_hsmm.py      # -> eda_out/reports/P7w_hsmm.md

Per test match: duration prior from the other six matches' true possessions; grid of (alpha, beta, gamma),
gamma scaling v9's per-frame switch cost charged at each possession start; the setting best on the other
six is used (nested). Dead ball v9's.
"""

import itertools
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
from mtah.hsmm import decode_hsmm, duration_prior  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p7l_dead_blocks import blocks, preds  # noqa: E402

GRID = list(itertools.product((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.0, 0.5, 1.0)))
OUT = os.path.join(ROOT, "eda_out", "reports", "P7w_hsmm.md")


def main():
    P = preds()
    n = len(P)
    S, N, base, nb = {}, {}, {}, {}
    for t, d in P.items():
        g = d["g"]
        dead = blocks((d["qh"] + d["qg"]) / 2, 0.5, 2, 2) & g
        alive = g & ~dead
        runs = pm.valid_runs(d["labels"])
        cost = G.SWITCH_HI * (1 - d["b"]) + G.SWITCH_LO * d["b"]
        c = np.where(g, decode(d["p"], alive, cost), pm.X).astype(np.int8)
        base[t] = pm.spell_scores(d["labels"], c, runs)["mean_f1"]
        nb[t] = len(pm.possession_spells(c, runs))
        lp = duration_prior([P[u]["labels"] for u in range(n) if u != t])
        for a, b, gm in GRID:
            c = np.where(g, decode_hsmm(d["p"], alive, lp, alpha=a, beta=b, cost=gm * cost), pm.X).astype(np.int8)
            S[(t, (a, b, gm))] = pm.spell_scores(d["labels"], c, runs)["mean_f1"]
            N[(t, (a, b, gm))] = len(pm.possession_spells(c, runs))
        print(d["name"], "v9", round(base[t], 3), "best", max((S[(t, k)], k) for k in GRID), flush=True)
    rows = []
    for t in range(n):
        o = [u for u in range(n) if u != t]
        k = max(GRID, key=lambda k: np.mean([S[(u, k)] for u in o]))
        rows.append({"match": P[t]["name"], "n_gt": len(pm.possession_spells(P[t]["labels"], pm.valid_runs(P[t]["labels"]))),
                     "v9": base[t], "n_v9": nb[t], "hsmm": S[(t, k)], "n_hsmm": N[(t, k)], "chosen": k})
    df = pd.DataFrame(rows)
    d = df.hsmm - df.v9
    top = sorted(GRID, key=lambda k: -np.mean([S[(t, k)] for t in range(n)]))[:6]
    lines = ["# P7w: süreli kod çözme (HSMM) v9'un Viterbi'sine karşı (iç içe, tam zincir, taşınan ölü toplu DFL)", "", "`analysis/p7w_hsmm.py`. Ayar (alpha, beta, gamma).", "",
             "| maç | gerçek poss. | v9 | HSMM | seçilen |", "|---|---|---|---|---|"]
    lines += ["| {} | {} | {:.3f} ({}) | {:.3f} ({}) | {} |".format(r["match"], r["n_gt"], r["v9"], r["n_v9"], r["hsmm"], r["n_hsmm"], r["chosen"]) for r in rows]
    lines += ["| ortalama | {:.0f} | {:.3f} ({:.0f}) | {:.3f} ({:.0f}) | |".format(df.n_gt.mean(), df.v9.mean(), df.n_v9.mean(), df.hsmm.mean(), df.n_hsmm.mean()), "",
              "Fark {:+.3f} ({:+.3f} .. {:+.3f}), {} / 7 maçta önde.".format(d.mean(), d.min(), d.max(), int((d > 0).sum())), "",
              "Bütün maçlarda en iyi 6: " + "; ".join("{} {:.3f}".format(k, np.mean([S[(t, k)] for t in range(n)])) for k in top)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
