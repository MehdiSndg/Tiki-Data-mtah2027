"""The night's candidates together in the full chain, on the carried-dead-ball DFL (as P6y carry_cmp). GPU.

    python analysis/p7f_combo_chain.py      # -> eda_out/reports/P7f_combo_chain.md

Per DFL test match (degraded with dead_carry_p = 0.25), trained on the other six (+ Metrica 1-2 for the
dead ball). v6: team + boundary GRUs on the old degradation, dead-ball v4 HistGB. swap: both GRUs also
trained on every match with the teams swapped (P7e). +deadmean: dead-ball probability the mean of the v4
HistGB and a GRU on the same features (P7c). +set: the team GRU with mtah.setgru's player set (P7d),
swapped too. +tta: P(home) also predicted on the test match swapped, 1 - that averaged in. 3 seeds.
"""

import glob
import os
import sys
from functools import partial

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
from mtah.setgru import player_set, set_net  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402
from p7e_team_swap import swap_teams  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P7f_combo_chain.md")
COLS = ["v6", "swap", "swap_deadmean", "swap_deadmean_tta", "swap_deadmean_set", "swap_deadmean_set_tta", "truth_dead_swap"]


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2)]
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    new = [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=0) for r, k in raws]
    nd = len(ids)
    old = old[:nd]
    so, sn = [swap_teams(m) for m in old], [swap_teams(m) for m in new[:nd]]
    To, Tso = [features(m) for m in old], [features(m) for m in so]
    Tn, Tsn = [features(m) for m in new[:nd]], [features(m) for m in sn]
    Po, Pso = [np.concatenate([t, player_set(m)], 1) for t, m in zip(To, old)], [np.concatenate([t, player_set(m)], 1) for t, m in zip(Tso, so)]
    Pn, Psn = [np.concatenate([t, player_set(m)], 1) for t, m in zip(Tn, new[:nd])], [np.concatenate([t, player_set(m)], 1) for t, m in zip(Tsn, sn)]
    Dn = [features_v2(m) for m in new]
    nft = To[0].shape[1]
    mk = partial(lambda nf, n0: set_net(n0), n0=nft)
    print("features done", flush=True)
    it = lambda m, F, y: (F, y, np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32), m["half"] > 0)
    team = lambda mm: (mm["labels"] == pm.H).astype(np.float32)
    bnd = lambda mm: G.boundary_target(mm["labels"])
    rows = []
    for t in range(nd):
        tr = [k for k in range(nd) if k != t]
        trd = tr + [nd, nd + 1]
        m = new[t]
        g = m["half"] > 0
        p0 = fit_predict([it(old[k], To[k], team(old[k])) for k in tr], (Tn[t], g))
        b0 = fit_predict([it(old[k], To[k], bnd(old[k])) for k in tr], (Tn[t], g))
        sw_team = [it(old[k], To[k], team(old[k])) for k in tr] + [it(so[k], Tso[k], team(so[k])) for k in tr]
        sw_bnd = [it(old[k], To[k], bnd(old[k])) for k in tr] + [it(so[k], Tso[k], bnd(so[k])) for k in tr]
        p1 = fit_predict(sw_team, (Tn[t], g))
        p1s = 1 - fit_predict(sw_team, (Tsn[t], g))
        b1 = fit_predict(sw_bnd, (Tn[t], g))
        set_team = [it(old[k], Po[k], team(old[k])) for k in tr] + [it(so[k], Pso[k], team(so[k])) for k in tr]
        p2 = fit_predict(set_team, (Pn[t], g), make_net=mk)
        p2s = 1 - fit_predict(set_team, (Psn[t], g), make_net=mk)
        c0, c1 = G.SWITCH_HI * (1 - b0) + G.SWITCH_LO * b0, G.SWITCH_HI * (1 - b1) + G.SWITCH_LO * b1
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
            pd.concat([Dn[k][new[k]["half"] > 0] for k in trd], ignore_index=True), np.concatenate([new[k]["labels"][new[k]["half"] > 0] == pm.D for k in trd]))
        qh = np.zeros(len(g))
        qh[g] = clf.predict_proba(Dn[t][g])[:, 1]
        qg = fit_predict([(Dn[k].values.astype(np.float32), (new[k]["labels"] == pm.D).astype(np.float32), (new[k]["half"] > 0).astype(np.float32), new[k]["half"] > 0) for k in trd],
                         (Dn[t].values.astype(np.float32), g))
        dh, dm = to_blocks(qh) & g, to_blocks((qh + qg) / 2) & g
        sc = lambda p, c, dead: pm.spell_scores(m["labels"], np.where(g, decode(p, g & ~dead, c), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"], "v6": sc(p0, c0, dh), "swap": sc(p1, c1, dh), "swap_deadmean": sc(p1, c1, dm),
             "swap_deadmean_tta": sc((p1 + p1s) / 2, c1, dm), "swap_deadmean_set": sc(p2, c1, dm),
             "swap_deadmean_set_tta": sc((p2 + p2s) / 2, c1, dm), "truth_dead_swap": sc(p1, c1, m["labels"] == pm.D)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P7f: gecenin adayları tam zincirde (taşınan ölü toplu DFL, maç dışarıda bırakarak, 3 tohum)", "", "`analysis/p7f_combo_chain.py`.", "",
             "| maç | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in COLS) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in COLS) + " |", ""]
    lines += ["{}: v6'ya fark {:+.3f} ({:+.3f} .. {:+.3f}), {} / 7 maçta önde.".format(c, (df[c] - df.v6).mean(), (df[c] - df.v6).min(), (df[c] - df.v6).max(), int((df[c] > df.v6).sum())) for c in COLS[1:-1]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
