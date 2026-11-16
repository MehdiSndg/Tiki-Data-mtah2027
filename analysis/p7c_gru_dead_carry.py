"""The dead ball from a GRU, on the carried-dead-ball degradation (P6v was before it). Full chain, DFL LOMO. GPU.

    python analysis/p7c_gru_dead_carry.py      # -> eda_out/reports/P7c_gru_dead_carry.md

As P6y carry_cmp (test on dead_carry_p = 0.25; team and boundary GRUs on the old degradation, 3 seeds, the
v5 switch cost): dead ball from the v4 HistGB (production: dead-ball v2 features, trained on the carried
degradation), from a GRU on the same features, from a GRU on those + the team features, the mean of the
HistGB and each GRU, and from the labels. Dead-ball models trained on the other six + Metrica 1-2, carried.
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

OUT = os.path.join(ROOT, "eda_out", "reports", "P7c_gru_dead_carry.md")


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2)]
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    new = [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=0) for r, k in raws]
    To, Tn = [features(m) for m in old], [features(m) for m in new]
    Dn = [features_v2(m) for m in new]
    Cn = [np.concatenate([D.values.astype(np.float32), T.astype(np.float32)], axis=1) for D, T in zip(Dn, Tn)]
    print("features done", flush=True)
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
        sc = lambda dead: pm.spell_scores(m["labels"], np.where(g, decode(p, g & ~dead, cost), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
            pd.concat([Dn[k][new[k]["half"] > 0] for k in trd], ignore_index=True), np.concatenate([new[k]["labels"][new[k]["half"] > 0] == pm.D for k in trd]))
        qh = np.zeros(len(g))
        qh[g] = clf.predict_proba(Dn[t][g])[:, 1]
        dtgt = lambda k: ((new[k]["labels"] == pm.D).astype(np.float32), (new[k]["half"] > 0).astype(np.float32), new[k]["half"] > 0)
        qg = gru_fit_predict([(Dn[k].values.astype(np.float32),) + dtgt(k) for k in trd], (Dn[t].values.astype(np.float32), g), seeds=G.SEEDS)
        qc = gru_fit_predict([(Cn[k],) + dtgt(k) for k in trd], (Cn[t], g), seeds=G.SEEDS)
        r = {"match": m["name"], "hgb": sc(to_blocks(qh) & g), "gru": sc(to_blocks(qg) & g), "gru_all": sc(to_blocks(qc) & g),
             "mean": sc(to_blocks((qh + qg) / 2) & g), "mean_all": sc(to_blocks((qh + qc) / 2) & g), "truth": sc(m["labels"] == pm.D)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["hgb", "gru", "gru_all", "mean", "mean_all", "truth"]
    lines = ["# P7c: ölü top GRU ile, taşınan ölü toplu DFL'de (tam zincir, v5 tarifinin takım kısmı, 3 tohum)", "", "`analysis/p7c_gru_dead_carry.py`.", "",
             "| maç | HistGB (v6) | GRU | GRU + takım özellikleri | ort. HistGB+GRU | ort. HistGB+GRU tüm | etiketten |", "|---|---|---|---|---|---|---|"]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", ""]
    lines += ["{}: HistGB'ye fark {:+.3f}, {} / {} maçta önde.".format(c, (df[c] - df.hgb).mean(), int((df[c] > df.hgb).sum()), len(df)) for c in cols[1:5]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
