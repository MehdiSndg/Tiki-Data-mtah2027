"""Dead-ball models on five degradation draws against three (v9). Full chain on the carried-dead-ball DFL. GPU.
(carried dead ball) instead of seed 0 alone. Full chain on the carried-dead-ball DFL, v8 recipe. GPU.

    python analysis/p7q_dead_five.py      # -> eda_out/reports/P7q_dead_five.md

Per DFL test match (degraded seed 0, dead_carry_p = 0.25): team + boundary GRUs with the team swap on the
old degradation (v7/v8), dead ball the mean of the v2-features HistGB and GRU (v8), both trained on the
other six + Metrica 1-2 with one degradation draw or with three.
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
from m1_dead_classifier import SEED as DSEED, to_blocks  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P7q_dead_five.md")
DRAWS = (0, 1, 2, 3, 4)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2)]
    nd = len(ids)
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws[:nd]]
    so = [swap_teams(m) for m in old]
    new = {s: [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=s) for r, k in raws] for s in DRAWS}
    To, Tso, Tn = [features(m) for m in old], [features(m) for m in so], [features(m) for m in new[0][:nd]]
    Dn = {s: [features_v2(m) for m in new[s]] for s in DRAWS}
    print("features done", flush=True)
    it = lambda m, F, y: (F, y, np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32), m["half"] > 0)
    team = lambda mm: (mm["labels"] == pm.H).astype(np.float32)
    bnd = lambda mm: G.boundary_target(mm["labels"])
    rows = []
    for t in range(nd):
        tr = [k for k in range(nd) if k != t]
        trd = tr + [nd, nd + 1]
        m = new[0][t]
        g = m["half"] > 0
        p = fit_predict([it(old[k], To[k], team(old[k])) for k in tr] + [it(so[k], Tso[k], team(so[k])) for k in tr], (Tn[t], g))
        b = fit_predict([it(old[k], To[k], bnd(old[k])) for k in tr] + [it(so[k], Tso[k], bnd(so[k])) for k in tr], (Tn[t], g))
        cost = G.SWITCH_HI * (1 - b) + G.SWITCH_LO * b
        sc = lambda dead: pm.spell_scores(m["labels"], np.where(g, decode(p, g & ~dead, cost), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"]}
        for tag, draws in (("one", (0, 1, 2)), ("three", DRAWS)):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
                pd.concat([Dn[s][k][new[s][k]["half"] > 0] for s in draws for k in trd], ignore_index=True),
                np.concatenate([new[s][k]["labels"][new[s][k]["half"] > 0] == pm.D for s in draws for k in trd]))
            qh = np.zeros(len(g))
            qh[g] = clf.predict_proba(Dn[0][t][g])[:, 1]
            qg = fit_predict([(Dn[s][k].values.astype(np.float32), (new[s][k]["labels"] == pm.D).astype(np.float32), (new[s][k]["half"] > 0).astype(np.float32),
                               new[s][k]["half"] > 0) for s in draws for k in trd], (Dn[0][t].values.astype(np.float32), g))
            r[tag] = sc(to_blocks((qh + qg) / 2) & g)
            r[tag + "_hgb"] = sc(to_blocks(qh) & g)
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["one_hgb", "one", "three_hgb", "three"]
    lines = ["# P7q: ölü top modelleri beş bozma çekilişiyle (üçe karşı) (taşınan ölü toplu DFL, tam zincir, v8 tarifi, 3 tohum)", "", "`analysis/p7q_dead_five.py`.", "",
             "| maç | HistGB, 3 | ort., 3 (v9) | HistGB, 5 | ort., 5 |", "|---|---|---|---|---|"]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", "",
              "5 çekiliş − 3 çekiliş (ort.): {:+.3f}, {} / 7 maçta önde.".format((df.three - df.one).mean(), int((df.three > df.one).sum()))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
