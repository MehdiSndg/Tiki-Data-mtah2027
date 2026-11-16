"""Confirmation with fresh seeds (3, 4, 5): v9 against v9 + the small positives of P7i, P7s and P7u together,
in the full chain on the carried-dead-ball DFL. GPU.

    python analysis/p7x_combo_fresh.py      # -> eda_out/reports/P7x_combo_fresh.md

Each of the three was within noise alone and picked among many tries, so the combination is checked on
seeds none of them were chosen with. combo: the team call the mean of the GRU and the TCN (p7u), both
trained on DFL + Metrica 1-3 (P7i), swapped, with short possessions weighted K = 2 (P7s); the boundary
GRU as v9 but on DFL + Metrica. Dead ball v9's (P7l's cache, seeds 0-2).
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
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402
from p7l_dead_blocks import blocks, preds  # noqa: E402
from p7s_short_weight import short_mask  # noqa: E402
from p7u_tcn import tcn  # noqa: E402

SEEDS = (3, 4, 5)
K = 2.0
OUT = os.path.join(ROOT, "eda_out", "reports", "P7x_combo_fresh.md")


def main():
    P = preds()
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    nd = len(ids)
    old = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    old += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    so = [swap_teams(m) for m in old]
    new = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25, dead_carry_p=0.25), seed=0) for i in ids]
    To, Tso, Tn = [features(m) for m in old], [features(m) for m in so], [features(m) for m in new]
    Sh = [short_mask(m["labels"]) for m in old]
    print("features done", flush=True)
    alive_w = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    team = lambda m: (m["labels"] == pm.H).astype(np.float32)
    rows = []
    for t in range(nd):
        d = P[t]
        g = d["g"]
        dead = blocks((d["qh"] + d["qg"]) / 2, 0.5, 2, 2) & g
        dk = [k for k in range(nd) if k != t]
        ak = [k for k in range(len(old)) if k != t]
        it = lambda k, F, m, w: (F, team(m), w, m["half"] > 0)
        base_t = [it(k, To[k], old[k], alive_w(old[k])) for k in dk] + [it(k, Tso[k], so[k], alive_w(so[k])) for k in dk]
        base_b = [(To[k], G.boundary_target(old[k]["labels"]), alive_w(old[k]), old[k]["half"] > 0) for k in dk] + \
                 [(Tso[k], G.boundary_target(so[k]["labels"]), alive_w(so[k]), so[k]["half"] > 0) for k in dk]
        wk = lambda k: alive_w(old[k]) * np.where(Sh[k], K, 1.0).astype(np.float32)
        combo_t = [it(k, To[k], old[k], wk(k)) for k in ak] + [it(k, Tso[k], so[k], wk(k)) for k in ak]
        combo_b = [(To[k], G.boundary_target(old[k]["labels"]), alive_w(old[k]), old[k]["half"] > 0) for k in ak] + \
                  [(Tso[k], G.boundary_target(so[k]["labels"]), alive_w(so[k]), so[k]["half"] > 0) for k in ak]
        sc = lambda p, b: pm.spell_scores(d["labels"], np.where(g, decode(p, g & ~dead, G.SWITCH_HI * (1 - b) + G.SWITCH_LO * b), pm.X).astype(np.int8),
                                          pm.valid_runs(d["labels"]))["mean_f1"]
        p0, b0 = fit_predict(base_t, (Tn[t], g), seeds=SEEDS), fit_predict(base_b, (Tn[t], g), seeds=SEEDS)
        pg = fit_predict(combo_t, (Tn[t], g), seeds=SEEDS)
        pt = fit_predict(combo_t, (Tn[t], g), make_net=tcn, seeds=SEEDS)
        b1 = fit_predict(combo_b, (Tn[t], g), seeds=SEEDS)
        r = {"match": d["name"], "v9": sc(p0, b0), "combo": sc((pg + pt) / 2, b1), "combo_gru": sc(pg, b1)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["v9", "combo_gru", "combo"]
    lines = ["# P7x: küçük artıların birleşimi, yeni tohumlarla (3, 4, 5; tam zincir, taşınan ölü toplu DFL)", "", "`analysis/p7x_combo_fresh.py`.", "",
             "| maç | v9 | GRU (Metrica + kısa ağırlık) | + TCN ortalaması |", "|---|---|---|---|"]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", ""]
    lines += ["{}: v9'a fark {:+.3f} ({:+.3f} .. {:+.3f}), {} / 7 maçta önde.".format(c, (df[c] - df.v9).mean(), (df[c] - df.v9).min(), (df[c] - df.v9).max(), int((df[c] > df.v9).sum())) for c in cols[1:]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
