"""Step 4: dead-ball features v2 (mtah.deadball) in the full chain, LOMO on degraded DFL.

    python analysis/p6j_dead_v2.py      # -> eda_out/reports/P6j_dead_v2.md

For each DFL test match: the production team model (mtah.teammodel features, trained on the other 6
DFL matches + Metrica 1-2, without the extra copies to keep it quick) and two dead-ball classifiers
trained on the same matches, m1's features and v2's. Both chains are scored with the official spell
metric; also the dead/alive balanced accuracy.
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
from m1_dead_classifier import SEED, features, to_blocks  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.teammodel import SWITCH, decode, features as team_feats, fit  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P6j_dead_v2.md")


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    ms += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2)]
    T, D1, D2 = [], [], []
    for m in ms:
        g = m["half"] > 0
        a = np.isin(m["labels"], (pm.H, pm.A))
        T.append((team_feats(m), a, (m["labels"] == pm.H).astype(int)))
        D1.append((features(m), g, m["labels"] == pm.D))
        D2.append((features_v2(m), g, m["labels"] == pm.D))
        print(m["name"], flush=True)
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ms)) if k != t]
        m = ms[t]
        g = m["half"] > 0
        p = fit([T[k][0][T[k][1]] for k in tr], [T[k][2][T[k][1]] for k in tr]).predict_proba(T[t][0])[:, 1]
        r = {"match": m["name"]}
        for tag, D in (("v1", D1), ("v2", D2)):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(
                pd.concat([D[k][0][D[k][1]] for k in tr], ignore_index=True), np.concatenate([D[k][2][D[k][1]] for k in tr]))
            q = np.zeros(len(g))
            q[g] = clf.predict_proba(D[t][0][g])[:, 1]
            dead = to_blocks(q) & g
            c = np.where(g, decode(p, g & ~dead, SWITCH), pm.X).astype(np.int8)
            r["spell_" + tag] = pm.spell_scores(m["labels"], c, pm.valid_runs(m["labels"]))["mean_f1"]
            r["bacc_" + tag] = balanced_accuracy_score(m["labels"][g] == pm.D, dead[g])
        truth = np.where(g, decode(p, np.isin(m["labels"], (pm.H, pm.A)), SWITCH), pm.X).astype(np.int8)
        r["spell_truth_dead"] = pm.spell_scores(m["labels"], truth, pm.valid_runs(m["labels"]))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P6j: ölü top özellikleri v2, tam zincirde (DFL, maç dışarıda bırakarak)", "", "`analysis/p6j_dead_v2.py`.", "",
             "| maç | spellF1 ölü top v1 | spellF1 ölü top v2 | ölü top etiketten | dengeli doğruluk v1 / v2 |", "|---|---|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} / {:.3f} |".format(r.match, r.spell_v1, r.spell_v2, r.spell_truth_dead, r.bacc_v1, r.bacc_v2) for r in df.itertuples()]
    d = df.spell_v2 - df.spell_v1
    lines += ["", "Ortalama: v1 {:.3f}, v2 {:.3f}, etiketten {:.3f}; fark {:+.3f} ({:+.3f} .. {:+.3f}, {} / {} maçta v2 önde).".format(
        df.spell_v1.mean(), df.spell_v2.mean(), df.spell_truth_dead.mean(), d.mean(), d.min(), d.max(), int((d > 0).sum()), len(d))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
