"""v9 against v9 + Metrica 1-3 in the team / boundary GRUs' training (P7i) and + a larger GRU (P7k), in the
full chain on the carried-dead-ball DFL. GPU.

    python analysis/p7m_final_combo.py [h256|l3]      # -> eda_out/reports/P7m_final_combo.md

Per DFL test match: team + boundary GRUs with the team swap trained on the other six DFL (v9) or on those
+ Metrica 1-3 (also swapped); production net or BIG; dead ball the v9 mean (HistGB + GRU, three draws).
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
from p7k_gru_size import net  # noqa: E402
from p7l_dead_blocks import blocks, preds  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P7m_final_combo.md")


def main(big="h256"):
    make = (lambda nf: net(nf, hidden=256)) if big == "h256" else (lambda nf: net(nf, layers=3))
    P = preds()                      # P7l's cache: v9 dead-ball probabilities per test match (and v9 team / boundary)
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    nd = len(ids)
    old = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    old += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    so = [swap_teams(m) for m in old]
    new = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25, dead_carry_p=0.25), seed=0) for i in ids]
    To, Tso, Tn = [features(m) for m in old], [features(m) for m in so], [features(m) for m in new]
    print("features done", flush=True)
    it = lambda m, F, y: (F, y, np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32), m["half"] > 0)
    team = lambda mm: (mm["labels"] == pm.H).astype(np.float32)
    bnd = lambda mm: G.boundary_target(mm["labels"])
    rows = []
    for t in range(nd):
        d = P[t]
        g = d["g"]
        dead = blocks((d["qh"] + d["qg"]) / 2, 0.5, 2, 2) & g
        sc = lambda p, b: pm.spell_scores(d["labels"], np.where(g, decode(p, g & ~dead, G.SWITCH_HI * (1 - b) + G.SWITCH_LO * b), pm.X).astype(np.int8),
                                          pm.valid_runs(d["labels"]))["mean_f1"]
        r = {"match": d["name"], "v9": sc(d["p"], d["b"])}
        for tag, ks in (("met", [k for k in range(len(old)) if k != t]),):
            tt = [it(old[k], To[k], team(old[k])) for k in ks] + [it(so[k], Tso[k], team(so[k])) for k in ks]
            tb = [it(old[k], To[k], bnd(old[k])) for k in ks] + [it(so[k], Tso[k], bnd(so[k])) for k in ks]
            r["met"] = sc(fit_predict(tt, (Tn[t], g)), fit_predict(tb, (Tn[t], g)))
            r["met_big"] = sc(fit_predict(tt, (Tn[t], g), make_net=make), fit_predict(tb, (Tn[t], g), make_net=make))
        dk = [k for k in range(nd) if k != t]
        tt = [it(old[k], To[k], team(old[k])) for k in dk] + [it(so[k], Tso[k], team(so[k])) for k in dk]
        tb = [it(old[k], To[k], bnd(old[k])) for k in dk] + [it(so[k], Tso[k], bnd(so[k])) for k in dk]
        r["big"] = sc(fit_predict(tt, (Tn[t], g), make_net=make), fit_predict(tb, (Tn[t], g), make_net=make))
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["v9", "met", "big", "met_big"]
    lines = ["# P7m: v9 + Metrica + büyük GRU ({}) tam zincirde (taşınan ölü toplu DFL, 3 tohum)".format(big), "", "`analysis/p7m_final_combo.py`.", "",
             "| maç | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", ""]
    lines += ["{}: v9'a fark {:+.3f} ({:+.3f} .. {:+.3f}), {} / 7 maçta önde.".format(c, (df[c] - df.v9).mean(), (df[c] - df.v9).min(), (df[c] - df.v9).max(), int((df[c] > df.v9).sum())) for c in cols[1:]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
