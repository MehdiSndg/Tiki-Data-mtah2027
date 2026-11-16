"""P9dg: do the ball-status Metrica labels (Mali's delivery, cache/team_in/mehmet_ali_topkara/external/metrica_<g>.npz:
dead where the ball track is missing >= 2 s, live where the ball is there) train a better dead-ball classifier than
ours (event-derived; 5-6 % of frames differ, ours calls live play with a ball dead and ball-less stretches live)?
HistGB on dead-ball v2 features, degraded copies (three draws) as v8; P9dc's feature cache.
  fold = one DFL match held out (train: 6 DFL + Metrica 1-3 with either label set) -> judged on DFL's own labels
  fold = one Metrica game held out (train: 7 DFL + the other two) -> judged on both label sets
Score: dead frame accuracy and spellF1 with the labels' team and the predicted dead ball (1.0 = perfect dead ball).

    python analysis/p9dg_metrica_labels.py [row step = 3]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
import p4_common as P4  # noqa: E402
from m1_dead_classifier import SEED, to_blocks  # noqa: E402
from p9db_pff_dead import oracle_team  # noqa: E402
from p9dc_dead_real import feats, sim_matches  # noqa: E402


def main(step=3):
    from sklearn.ensemble import HistGradientBoostingClassifier
    data = {}
    for key, make in sim_matches():
        F, lab, _ = feats(key, make)
        _, name, draw = key.split("_", 2) if key.count("_") == 2 else (None, key[4:].rsplit("_", 1)[0], key.rsplit("_", 1)[1])
        name, draw = key[4:].rsplit("_", 1)
        alt = lab
        if name.startswith("metrica"):
            z = np.load(os.path.join(ROOT, "cache", "team_in", "mehmet_ali_topkara", "external", "metrica_{}.npz".format(name[-1])))["labels"]
            g = np.ones(len(lab), bool)
            if len(z) == len(lab) - 1:                          # our grid has one separator frame between the halves
                sep = int(np.flatnonzero(np.diff(np.r_[0, (lab != pm.X).astype(int)]) != 0)[0]) if False else None
                from mtah.external import from_metrica_sample
                g = from_metrica_sample(int(name[-1]))["half"] > 0
            alt = np.full(len(lab), pm.X, np.int8)
            alt[g] = z
        data[(name, int(draw))] = (F, lab, alt)
    names = sorted({k[0] for k in data})
    fit = lambda X, y: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(X, y)

    def table(train, which):
        X, y = [], []
        for (n, d), (F, lab, alt) in data.items():
            if n not in train:
                continue
            L = alt if which == "ball" else lab
            k = np.flatnonzero(L != pm.X)[::step]
            X.append(F[k])
            y.append(L[k] == pm.D)
        return np.concatenate(X), np.concatenate(y)

    R = {}
    for t in names:
        F, lab, alt = data[(t, 0)]
        for which in ("ours", "ball"):
            clf = fit(*table([n for n in names if n != t], which))
            for judge, L in (("own labels", lab),) + ((("ball labels", alt),) if t.startswith("metrica") else ()):
                g = L != pm.X
                p = np.zeros(len(g))
                p[g] = clf.predict_proba(F[g])[:, 1]
                dead = to_blocks(p) & g
                f1 = pm.spell_scores(L, oracle_team(L, dead, g), pm.valid_runs(L))["mean_f1"]
                R.setdefault(("metrica" if t.startswith("metrica") else "DFL", judge, which), []).append((np.mean(dead[g] == (L[g] == pm.D)), f1))
                print(t, "trained on", which, "| judge", judge, "| accuracy {:.4f} spellF1 {:.4f}".format(*R[("metrica" if t.startswith("metrica") else "DFL", judge, which)][-1]), flush=True)
    print()
    for (src, judge, which), v in sorted(R.items()):
        print("MEAN", src, "| judge", judge, "| trained on", which, "| accuracy {:.4f} spellF1 {:.4f} (n {})".format(*np.mean(v, 0), len(v)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
