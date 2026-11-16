"""The v10 recipe in the full chain on Metrica games 1-3 (each left out in turn). GPU.

    python analysis/p8a_metrica_chain.py [GAME ...]      # -> eda_out/reports/P8a_metrica_chain.md (default 1 2 3)

A second source next to DFL for the headline number. Per left-out Metrica game: team GRU and TCN (DFL +
the other two Metrica games, swapped, short possessions x2) averaged, boundary GRU (same data, no weight),
dead ball the v9/v10 mean of HistGB and GRU on the dead-ball v2 features trained on the 7 DFL + the other
Metrica games with three degradation draws of the carried dead ball. Test degraded as production with the
carried dead ball. Also v9's team model (GRU, DFL only, swapped) for comparison.
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

DRAWS = (0, 1, 2)
OUT = os.path.join(ROOT, "eda_out", "reports", "P8a_metrica_chain.md")


def main(*games):
    games = [int(x) for x in games] or [1, 2, 3]
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    nd = len(ids)
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    so = [swap_teams(m) for m in old]
    new = {s: [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=s) for r, k in raws] for s in DRAWS}
    To, Tso = [features(m) for m in old], [features(m) for m in so]
    Dn = {s: [features_v2(m) for m in new[s]] for s in DRAWS}
    Sh = [G.short_mask(m["labels"]) for m in old]
    print("features done", flush=True)
    alive_w = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    team = lambda m: (m["labels"] == pm.H).astype(np.float32)
    rows = []
    for t in [nd + gm - 1 for gm in games]:
        m = new[0][t]
        g = m["half"] > 0
        Tt = features(m)
        ks = [k for k in range(len(old)) if k != t]
        wk = lambda k: alive_w(old[k]) * np.where(Sh[k], 2.0, 1.0).astype(np.float32)
        tt = [(To[k], team(old[k]), wk(k), old[k]["half"] > 0) for k in ks] + [(Tso[k], team(so[k]), wk(k), so[k]["half"] > 0) for k in ks]
        tb = [(To[k], G.boundary_target(old[k]["labels"]), alive_w(old[k]), old[k]["half"] > 0) for k in ks] + \
             [(Tso[k], G.boundary_target(so[k]["labels"]), alive_w(so[k]), so[k]["half"] > 0) for k in ks]
        v9t = [(To[k], team(old[k]), alive_w(old[k]), old[k]["half"] > 0) for k in range(nd)] + \
              [(Tso[k], team(so[k]), alive_w(so[k]), so[k]["half"] > 0) for k in range(nd)]
        v9b = [(To[k], G.boundary_target(old[k]["labels"]), alive_w(old[k]), old[k]["half"] > 0) for k in range(nd)] + \
              [(Tso[k], G.boundary_target(so[k]["labels"]), alive_w(so[k]), so[k]["half"] > 0) for k in range(nd)]
        p10 = (fit_predict(tt, (Tt, g)) + fit_predict(tt, (Tt, g), make_net=G._tcn)) / 2
        b10, p9, b9 = fit_predict(tb, (Tt, g)), fit_predict(v9t, (Tt, g)), fit_predict(v9b, (Tt, g))
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
            pd.concat([Dn[s][k][new[s][k]["half"] > 0] for s in DRAWS for k in ks], ignore_index=True),
            np.concatenate([new[s][k]["labels"][new[s][k]["half"] > 0] == pm.D for s in DRAWS for k in ks]))
        qh = np.zeros(len(g))
        qh[g] = clf.predict_proba(Dn[0][t][g])[:, 1]
        qg = fit_predict([(Dn[s][k].values.astype(np.float32), (new[s][k]["labels"] == pm.D).astype(np.float32), (new[s][k]["half"] > 0).astype(np.float32),
                           new[s][k]["half"] > 0) for s in DRAWS for k in ks], (Dn[0][t].values.astype(np.float32), g))
        dead = to_blocks((qh + qg) / 2) & g
        sc = lambda p, b, dd: pm.spell_scores(m["labels"], np.where(g, decode(p, g & ~dd, G.SWITCH_HI * (1 - b) + G.SWITCH_LO * b), pm.X).astype(np.int8),
                                              pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"], "v9": sc(p9, b9, dead), "v10": sc(p10, b10, dead), "v10_team_only": sc(p10, b10, m["labels"] == pm.D)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P8a: v10 tarifi Metrica 1–3'te tam zincirde (her maç sırayla dışarıda, 3 tohum)", "", "`analysis/p8a_metrica_chain.py`.", "",
             "| maç | v9 takım modeli | v10 | v10, ölü top etiketten |", "|---|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} | {:.3f} |".format(r["match"], r["v9"], r["v10"], r["v10_team_only"]) for r in rows]
    lines += ["| ortalama | {:.3f} | {:.3f} | {:.3f} |".format(df.v9.mean(), df.v10.mean(), df.v10_team_only.mean())]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
