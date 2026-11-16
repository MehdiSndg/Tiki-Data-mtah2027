"""The v5 recipe in the full chain on DFL LOMO: team GRU + boundary GRU (3 seeds each) + dead-ball v2. GPU.

    python analysis/p6y_v5_full_chain.py [carry]     # -> eda_out/reports/P6y_v5_full_chain[_carry].md

carry: every match degraded with the carried dead ball (p4_common.CARRY_RATES), for training and test.
carry_cmp: test on the carried-dead-ball degradation, the GRUs trained on the old degradation (as in
production), the dead-ball classifier on the old (v5) or the new (v6) one.

Per DFL test match, everything trained on the other six (+ Metrica 1-2 for the dead-ball classifier, as
P6j): the team GRU and the boundary GRU (mtah.grumodel recipe, seeds 0-2 averaged), the dead-ball v2
HistGB and its blocks; decoded with the flat SWITCH and with the per-frame cost of mtah.grumodel
(SWITCH_HI, SWITCH_LO). Both team-only (dead ball from the labels) and full chain.
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
from mtah.teammodel import decode, features  # noqa: E402
from p6v_gru_dead import gru_fit_predict  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P6y_v5_full_chain.md")


def carry_cmp():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2)]
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    new = [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=0) for r, k in raws]
    To, Tn = [features(m) for m in old], [features(m) for m in new]
    Do, Dn = [features_v2(m) for m in old], [features_v2(m) for m in new]
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        trd = tr + [len(ids), len(ids) + 1]
        m = new[t]
        g = m["half"] > 0
        lab = lambda k: old[k]["labels"]
        al = lambda k: np.isin(lab(k), (pm.H, pm.A)).astype(np.float32)
        p = gru_fit_predict([(To[k], (lab(k) == pm.H).astype(np.float32), al(k), old[k]["half"] > 0) for k in tr], (Tn[t], g), seeds=G.SEEDS)
        b = gru_fit_predict([(To[k], G.boundary_target(lab(k)), al(k), old[k]["half"] > 0) for k in tr], (Tn[t], g), seeds=G.SEEDS)
        cost = G.SWITCH_HI * (1 - b) + G.SWITCH_LO * b
        sc = lambda alive: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, cost), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"], "team": sc(np.isin(m["labels"], (pm.H, pm.A)))}
        for tag, D, src in (("v5", Do, old), ("v6", Dn, new)):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
                pd.concat([D[k][src[k]["half"] > 0] for k in trd], ignore_index=True), np.concatenate([src[k]["labels"][src[k]["half"] > 0] == pm.D for k in trd]))
            q = np.zeros(len(g))
            q[g] = clf.predict_proba(Dn[t][g])[:, 1]
            r[tag] = sc(g & ~(to_blocks(q) & g))
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P6y carry_cmp: taşınan ölü toplu DFL'de v5 ve v6 (GRU'lar eski bozmayla, 3 tohum)", "", "`analysis/p6y_v5_full_chain.py carry_cmp`.", "",
             "| maç | takım (ölü GT'den) | tam zincir v5 | tam zincir v6 |", "|---|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} | {:.3f} |".format(r["match"], r["team"], r["v5"], r["v6"]) for r in rows]
    lines += ["| ortalama | {:.3f} | {:.3f} | {:.3f} |".format(df.team.mean(), df.v5.mean(), df.v6.mean())]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT.replace(".md", "_carry_cmp.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(mode=""):
    if mode == "carry_cmp":
        return carry_cmp()
    from sklearn.ensemble import HistGradientBoostingClassifier
    extra = {"dead_carry_p": 0.25} if mode == "carry" else {}
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25, **extra), seed=0) for i in ids]
    ms += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25, **extra), seed=0) for g in (1, 2)]
    T = [features(m) for m in ms]
    DF = [features_v2(m) for m in ms]
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        trd = tr + [len(ids), len(ids) + 1]
        m = ms[t]
        g = m["half"] > 0
        lab = lambda k: ms[k]["labels"]
        al = lambda k: np.isin(lab(k), (pm.H, pm.A)).astype(np.float32)
        p = gru_fit_predict([(T[k], (lab(k) == pm.H).astype(np.float32), al(k), ms[k]["half"] > 0) for k in tr], (T[t], g), seeds=G.SEEDS)
        b = gru_fit_predict([(T[k], G.boundary_target(lab(k)), al(k), ms[k]["half"] > 0) for k in tr], (T[t], g), seeds=G.SEEDS)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
            pd.concat([DF[k][ms[k]["half"] > 0] for k in trd], ignore_index=True), np.concatenate([lab(k)[ms[k]["half"] > 0] == pm.D for k in trd]))
        q = np.zeros(len(g))
        q[g] = clf.predict_proba(DF[t][g])[:, 1]
        alive_clf = g & ~(to_blocks(q) & g)
        alive_true = np.isin(m["labels"], (pm.H, pm.A))
        cost = G.SWITCH_HI * (1 - b) + G.SWITCH_LO * b
        sc = lambda alive, c: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, c), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        rows.append({"match": m["name"], "team_flat": sc(alive_true, G.SWITCH), "team_v5": sc(alive_true, cost),
                     "chain_flat": sc(alive_clf, G.SWITCH), "chain_v5": sc(alive_clf, cost)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P6y: v5 tarifi tam zincirde (DFL, maç dışarıda bırakarak, 3 tohum)", "", "`analysis/p6y_v5_full_chain.py`.", "",
             "| maç | takım, sabit ceza | takım, v5 | tam zincir, sabit | tam zincir, v5 |", "|---|---|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(r["match"], r["team_flat"], r["team_v5"], r["chain_flat"], r["chain_v5"]) for r in rows]
    lines += ["| ortalama | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(df.team_flat.mean(), df.team_v5.mean(), df.chain_flat.mean(), df.chain_v5.mean())]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT.replace(".md", "_carry.md") if mode == "carry" else OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
