"""P9dc: a dead-ball classifier trained on real tracking (PFF World Cup broadcast tracking; P9db: v8's model, trained
on degraded DFL + Metrica only, caps spellF1 at ~0.85 on real tracking even with a perfect team call).
Arms (HistGradientBoosting on dead-ball v2 features, as v8's classifier):
  sim        degraded DFL + Metrica 1-3, three draws (v8's classifier)
  real       PFF training matches (ball as given; and a second copy with only the seen ball)
  sim+real   both
Judges, never trained on: the held-out PFF matches (the last <n test> ids) and SoccerTrack matches (degraded as P9bw).
Per arm: dead frame accuracy, recall, precision, and spellF1 with the labels' team and the predicted dead ball
(to_blocks of the probability; 1.0 = perfect dead ball). Features cached in cache/p9dc/.

    python analysis/p9dc_dead_real.py [n test = 4] [soccertrack ids = 118577,117093,128057,118576] [row step = 3]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
import p4_common as P4  # noqa: E402
from m1_dead_classifier import SEED, to_blocks  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample, from_soccertrack  # noqa: E402
from mtah.pff import from_pff, pff_ids  # noqa: E402
from p9db_pff_dead import oracle_team  # noqa: E402

FC = os.path.join(ROOT, "cache", "p9dc")


def feats(key, make):
    """(features float32, labels, columns) of a match, cached."""
    path = os.path.join(FC, key + ".npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as d:
            return d["X"], d["lab"], [str(c) for c in d["cols"]]
    m = make()
    F = features_v2(m)
    os.makedirs(FC, exist_ok=True)
    np.savez_compressed(path, X=F.values.astype(np.float32), lab=m["labels"], cols=np.array(F.columns, "U"))
    return F.values.astype(np.float32), m["labels"], list(F.columns)


def sim_matches():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    for d in P4.DRAWS_V6:
        for i in ids:
            yield "sim_{}_{}".format(i, d), (lambda i=i, d=d: degrade(from_dfl(i), dict(rates_for(i), **P4.CARRY_RATES), seed=d))
        for gm in (1, 2, 3):
            yield "sim_metrica{}_{}".format(gm, d), (lambda gm=gm, d=d: degrade(from_metrica_sample(gm), dict(rates_for("metrica"), **P4.CARRY_RATES), seed=d))


def table(items, step):
    X, y = [], []
    for key, make in items:
        F, lab, _ = feats(key, make)
        k = np.flatnonzero(lab != pm.X)[::step]
        X.append(F[k])
        y.append(lab[k] == pm.D)
        print("  features", key, len(k), flush=True)
    return np.concatenate(X), np.concatenate(y)


def main(ntest=4, st="118577,117093,128057,118576", step=3):
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = pff_ids()
    tr, te = ids[:-ntest], ids[-ntest:]
    print("PFF train", len(tr), "test", te, flush=True)
    sim = table(list(sim_matches()), step)
    real = table([("pff_{}_{}".format(i, b), (lambda i=i, b=b: from_pff(i, ball=b))) for i in tr for b in ("all", "visible")], step)
    fit = lambda X, y: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(X, y)
    arms = {"sim": fit(*sim), "real": fit(*real), "sim+real": fit(np.concatenate([sim[0], real[0]]), np.concatenate([sim[1], real[1]]))}
    judges = [("pff " + i, "pff_{}_all".format(i), (lambda i=i: from_pff(i))) for i in te]
    judges += [("st " + i, "st_{}_0".format(i), (lambda i=i: (lambda m0: degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0))(from_soccertrack(i, merge_s=1.6))))
               for i in st.split(",") if i]
    R = {a: [] for a in arms}
    for name, key, make in judges:
        F, lab, _ = feats(key, make)
        g = lab != pm.X
        truth = lab == pm.D
        for a, clf in arms.items():
            p = np.zeros(len(g))
            p[g] = clf.predict_proba(F[g])[:, 1]
            dead = to_blocks(p) & g
            tp = np.sum(dead & truth)
            f1 = pm.spell_scores(lab, oracle_team(lab, dead, g), pm.valid_runs(lab))["mean_f1"]
            R[a].append((name[:3] == "pff", np.mean(dead[g] == truth[g]), tp / max(truth.sum(), 1), tp / max(dead.sum(), 1), f1))
            print(name, a, "accuracy {:.3f} recall {:.3f} precision {:.3f} spellF1 with the labels' team {:.4f}".format(*R[a][-1][1:]), flush=True)
    for a, v in R.items():
        v = np.array(v, float)
        for kind, k in (("PFF", v[:, 0] == 1), ("SoccerTrack", v[:, 0] == 0)):
            if k.any():
                print("MEAN", a, kind, "accuracy {:.3f} recall {:.3f} precision {:.3f} spellF1 {:.4f}".format(*v[k][:, 1:].mean(0)))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 4, a[1] if len(a) > 1 else "118577,117093,128057,118576", int(a[2]) if len(a) > 2 else 3)
