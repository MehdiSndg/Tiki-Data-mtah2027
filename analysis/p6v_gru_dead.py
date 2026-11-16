"""The dead ball from a GRU too: full chain with a GRU dead-ball model vs the HistGB one. LOMO on DFL. GPU.

    python analysis/p6v_gru_dead.py      # -> eda_out/reports/P6v_gru_dead.md

Per DFL test match, trained on the other six (+ Metrica 1-2 for the dead-ball models, as P6j): the team
GRU (mtah.grumodel, 2 seeds) as P6s; dead ball either the dead-ball v2 HistGB (mtah.deadball features) or
a GRU on the same features (mtah.grumodel.train with the dead target on all graded frames, 2 seeds).
Both dead-ball probabilities go through m1's to_blocks; then the Viterbi at the GRU switch cost.
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

SEEDS = (0, 1)
OUT = os.path.join(ROOT, "eda_out", "reports", "P6v_gru_dead.md")


def gru_fit_predict(train, test, seeds=SEEDS):
    """train: list of (F, y, w, g); test: (F, g). Standardised on train; mean P over seeds."""
    allF = np.concatenate([d[0] for d in train])
    mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
    data = [(mod.norm(F), y, w, g) for F, y, w, g in train]
    nets = [G.train(data, s, "cuda") for s in seeds]
    return np.mean([G._predict_one(n, mod.norm(test[0]), test[1], "cuda") for n in nets], axis=0)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    ms += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2)]
    T = [features(m) for m in ms]
    DF = [features_v2(m) for m in ms]
    print("features done", flush=True)
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        trd = tr + [len(ids), len(ids) + 1]
        m = ms[t]
        g = m["half"] > 0
        lab = lambda k: ms[k]["labels"]
        pt = gru_fit_predict([(T[k], (lab(k) == pm.H).astype(np.float32), np.isin(lab(k), (pm.H, pm.A)).astype(np.float32), ms[k]["half"] > 0) for k in tr],
                             (T[t], g))
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
            pd.concat([DF[k][ms[k]["half"] > 0] for k in trd], ignore_index=True), np.concatenate([lab(k)[ms[k]["half"] > 0] == pm.D for k in trd]))
        qb = np.zeros(len(g))
        qb[g] = clf.predict_proba(DF[t][g])[:, 1]
        qg = gru_fit_predict([(DF[k].values.astype(np.float32), (lab(k) == pm.D).astype(np.float32), (ms[k]["half"] > 0).astype(np.float32), ms[k]["half"] > 0) for k in trd],
                             (DF[t].values.astype(np.float32), g))
        sc = lambda dead: pm.spell_scores(m["labels"], np.where(g, decode(pt, g & ~dead, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"], "dead_hgb": sc(to_blocks(qb) & g), "dead_gru": sc(to_blocks(qg) & g), "dead_mean": sc(to_blocks((qb + qg) / 2) & g),
             "dead_truth": sc(m["labels"] == pm.D)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P6v: ölü top GRU ile (DFL, maç dışarıda bırakarak, tam zincir, takım GRU)", "", "`analysis/p6v_gru_dead.py`.", "",
             "| maç | ölü top HistGB | ölü top GRU | ortalama | etiketten |", "|---|---|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(r["match"], r["dead_hgb"], r["dead_gru"], r["dead_mean"], r["dead_truth"]) for r in rows]
    lines += ["", "Ortalama: HistGB {:.3f}, GRU {:.3f}, ortalama {:.3f}, etiketten {:.3f}.".format(
        df.dead_hgb.mean(), df.dead_gru.mean(), df.dead_mean.mean(), df.dead_truth.mean())]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
