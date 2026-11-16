"""How much of the dead ball should the training degradation carry? The dead-ball HistGB (dead-ball v2 features, three
degradation draws, v9's recipe) trained with dead_carry_p in TRAIN_P and tested on matches degraded with each test
carry in turn, DFL and Metrica 1-3 leave one match out. The team side comes from the labels (each alive frame keeps
its true team; frames called alive inside a true dead block take the neighbouring true team), so the score isolates the
dead ball. CPU (research report item 5: 0.25 was set from a ~13% visibility that was likely measured on our own dead mask,
while 66% of public dead frames show the ball).

    python analysis/p9d_dead_carry_sweep.py <test_carry>      # -> eda_out/reports/P9d_dead_carry_<test_carry>.md
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
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402

TRAIN_P = (0.25, 0.5, 0.75)
DRAWS = (0, 1, 2)


def oracle_team(labels, dead):
    """labels with D where dead is predicted and, where a true dead frame is called alive, the nearest true team."""
    team = pd.Series(np.where(np.isin(labels, (pm.H, pm.A)), labels, np.nan)).ffill().bfill().to_numpy()
    out = np.where(np.isin(labels, (pm.H, pm.A)), labels, team)
    out = np.where(labels == pm.X, pm.X, out)
    return np.where(dead, pm.D, out).astype(np.int8)


def main(test_carry):
    from sklearn.ensemble import HistGradientBoostingClassifier
    test_carry = float(test_carry)
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    tests = [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=test_carry), seed=0) for r, k in raws]
    Dt = [features_v2(m) for m in tests]
    rows = []
    for p in TRAIN_P:
        tr = {s: [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=p), seed=s) for r, k in raws] for s in DRAWS}
        Dn = {s: [features_v2(m) for m in tr[s]] for s in DRAWS}
        print("features done", p, flush=True)
        for t, m in enumerate(tests):
            ks = [k for k in range(len(raws)) if k != t]
            g = m["half"] > 0
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
                pd.concat([Dn[s][k][tr[s][k]["half"] > 0] for s in DRAWS for k in ks], ignore_index=True),
                np.concatenate([tr[s][k]["labels"][tr[s][k]["half"] > 0] == pm.D for s in DRAWS for k in ks]))
            q = np.zeros(len(g))
            q[g] = clf.predict_proba(Dt[t][g])[:, 1]
            dead = to_blocks(q) & g
            sc = pm.spell_scores(m["labels"], oracle_team(m["labels"], dead), pm.valid_runs(m["labels"]))["mean_f1"]
            r = {"match": m["name"], "train_p": p, "spellF1": sc, "dead_share": float(dead[g].mean()), "true_dead_share": float((m["labels"][g] == pm.D).mean())}
            rows.append(r)
            print(r, flush=True)
    df = pd.DataFrame(rows)
    piv = df.pivot(index="match", columns="train_p", values="spellF1")
    lines = ["# P9d: ölü top eğitim taşıma oranı, test taşıması {} (HistGB, takım etiketten)".format(test_carry), "",
             "`analysis/p9d_dead_carry_sweep.py {}`.".format(test_carry), "", "| maç | " + " | ".join("eğitim {}".format(p) for p in TRAIN_P) + " |",
             "|---|" + "---|" * len(TRAIN_P)]
    lines += ["| {} | ".format(i) + " | ".join("{:.3f}".format(piv.loc[i, p]) for p in TRAIN_P) + " |" for i in piv.index]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(piv[p].mean()) for p in TRAIN_P) + " |", ""]
    lines += ["eğitim {} − 0.25: {:+.3f}, {} / {} maçta önde.".format(p, (piv[p] - piv[0.25]).mean(), int((piv[p] > piv[0.25]).sum()), len(piv)) for p in TRAIN_P[1:]]
    out = os.path.join(ROOT, "eda_out", "reports", "P9d_dead_carry_{}.md".format(test_carry))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
