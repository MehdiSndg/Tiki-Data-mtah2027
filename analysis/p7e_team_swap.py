"""Team-swap augmentation for the team GRU: every training match also given with home and away swapped
(labels H <-> A, slot teams, keepers, attack sign). DFL LOMO, team call only, 3 seeds. GPU.

    python analysis/p7e_team_swap.py      # -> eda_out/reports/P7e_team_swap.md

The model's target is P(home); nothing about possession depends on which team is home, so the swapped
match is as real as the original and doubles the training data. Test on the original matches only.
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
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P7e_team_swap.md")


def swap_teams(m):
    s = dict(m)
    lab = m["labels"].copy()
    lab[m["labels"] == pm.H], lab[m["labels"] == pm.A] = pm.A, pm.H
    s["labels"] = lab
    s["slot_team"] = (1 - np.asarray(m["slot_team"])).astype(np.int8)
    s["gk"] = {0: m["gk"][1], 1: m["gk"][0]}
    s["sign"] = {h: -v for h, v in m["sign"].items()}
    s["name"] = m["name"] + "+swap"
    return s


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    sw = [swap_teams(m) for m in ms]
    T, Ts = [features(m) for m in ms], [features(m) for m in sw]
    print("features done", flush=True)
    it = lambda m, F, y: (F, y, np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32), m["half"] > 0)
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        m = ms[t]
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        team = lambda mm: (mm["labels"] == pm.H).astype(np.float32)
        bnd = lambda mm: G.boundary_target(mm["labels"])
        base_t = [it(ms[k], T[k], team(ms[k])) for k in tr]
        aug_t = base_t + [it(sw[k], Ts[k], team(sw[k])) for k in tr]
        base_b = [it(ms[k], T[k], bnd(ms[k])) for k in tr]
        aug_b = base_b + [it(sw[k], Ts[k], bnd(sw[k])) for k in tr]
        p0, p1 = fit_predict(base_t, (T[t], g)), fit_predict(aug_t, (T[t], g))
        b0, b1 = fit_predict(base_b, (T[t], g)), fit_predict(aug_b, (T[t], g))
        c0, c1 = G.SWITCH_HI * (1 - b0) + G.SWITCH_LO * b0, G.SWITCH_HI * (1 - b1) + G.SWITCH_LO * b1
        sc = lambda p, c: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, c), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"], "base": sc(p0, G.SWITCH), "swap": sc(p1, G.SWITCH), "base_b": sc(p0, c0), "swap_b": sc(p1, c1)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["base", "swap", "base_b", "swap_b"]
    lines = ["# P7e: takım değiştirme ile veri çoğaltma (DFL, maç dışarıda bırakarak, 3 tohum, sadece takım kararı)", "", "`analysis/p7e_team_swap.py`.", "",
             "| maç | GRU | GRU + takım değişimi | GRU + sınır (v5) | ikisi de takım değişimiyle |", "|---|---|---|---|---|"]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", "",
              "Sabit cezada {:+.3f} ({} / 7); sınırla {:+.3f} ({} / 7).".format((df.swap - df.base).mean(), int((df.swap > df.base).sum()),
                                                                             (df.swap_b - df.base_b).mean(), int((df.swap_b > df.base_b).sum()))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
