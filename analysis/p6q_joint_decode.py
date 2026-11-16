"""Dead ball and team decided together (mtah.teammodel.decode3) vs the current two steps, LOMO on DFL.

    python analysis/p6q_joint_decode.py      # -> eda_out/reports/P6q_joint_decode.md

For each DFL test match the production team features' model and the dead-ball v2 classifier are
trained on the other six DFL matches + Metrica 1-2 (as P6j, no extra copies). Two-step: the classifier's
blocks (m1 to_blocks) then the two-state Viterbi. Joint: the three-state Viterbi on the per-frame dead
probability, for several dead-ball switch costs; the cost for each test match is the one with the best
mean over the other six test matches.
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
from m1_dead_classifier import SEED, to_blocks  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.teammodel import SWITCH, decode, decode3, features as team_feats, fit  # noqa: E402

COSTS = (2.0, 4.0, 8.0, 16.0, 32.0)
OUT = os.path.join(ROOT, "eda_out", "reports", "P6q_joint_decode.md")


def spell(m, c):
    return pm.spell_scores(m["labels"], c, pm.valid_runs(m["labels"]))["mean_f1"]


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    ms += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2)]
    T, D = [], []
    for m in ms:
        g = m["half"] > 0
        T.append((team_feats(m), np.isin(m["labels"], (pm.H, pm.A)), (m["labels"] == pm.H).astype(int)))
        D.append((features_v2(m), g, m["labels"] == pm.D))
        print(m["name"], flush=True)
    res = {}
    for t in range(len(ids)):
        tr = [k for k in range(len(ms)) if k != t]
        m = ms[t]
        g = m["half"] > 0
        p = fit([T[k][0][T[k][1]] for k in tr], [T[k][2][T[k][1]] for k in tr]).predict_proba(T[t][0])[:, 1]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(
            pd.concat([D[k][0][D[k][1]] for k in tr], ignore_index=True), np.concatenate([D[k][2][D[k][1]] for k in tr]))
        q = np.zeros(len(g))
        q[g] = clf.predict_proba(D[t][0][g])[:, 1]
        dead = to_blocks(q) & g
        r = {"two_step": spell(m, np.where(g, decode(p, g & ~dead, SWITCH), pm.X).astype(np.int8))}
        qs = pd.Series(q).rolling(25, center=True, min_periods=1).mean().values
        for c in COSTS:
            r[c] = spell(m, decode3(p, qs, g, SWITCH, c))
        res[m["name"]] = r
        print(m["name"], r, flush=True)
    df = pd.DataFrame(res).T
    nested = []
    for name in df.index:
        others = df.drop(index=name)
        c = max(COSTS, key=lambda x: others[x].mean())
        nested.append((name, c, df.loc[name, c]))
    lines = ["# P6q: ölü top ve takım birlikte çözülünce (DFL, maç dışarıda bırakarak)", "", "`analysis/p6q_joint_decode.py`.", "",
             "| maç | iki adım | " + " | ".join("birlikte, ceza {:g}".format(c) for c in COSTS) + " | iç seçim |",
             "|---|---|" + "---|" * len(COSTS) + "---|"]
    for (name, c, v) in nested:
        r = df.loc[name]
        lines.append("| {} | {:.3f} | ".format(name, r["two_step"]) + " | ".join("{:.3f}".format(r[x]) for x in COSTS) + " | {:.3f} ({:g}) |".format(v, c))
    nv = np.array([v for _, _, v in nested])
    d = nv - df["two_step"].values
    lines += ["", "Ortalama: iki adım {:.3f}, birlikte (iç seçim) {:.3f}; fark {:+.3f} ({:+.3f} .. {:+.3f}, {} / {} maçta birlikte önde).".format(
        df["two_step"].mean(), nv.mean(), d.mean(), d.min(), d.max(), int((d > 0).sum()), len(d))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
