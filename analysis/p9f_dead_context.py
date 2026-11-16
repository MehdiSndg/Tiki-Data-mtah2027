"""Long-horizon context for the dead ball (research report item 12; Lang et al. 2022 reach stoppage F1 79.8 at IoU 0.5 from
players alone with context up to +-80 s, ours looks at most 6 s): the dead-ball HistGB (v9 recipe: three degradation
draws, carry 0.25) on features_v2 against features_v3 (plus mtah.deadball.context of player speed, standing share and
ball seen over 4 / 20 / 40 / 80 s). DFL and Metrica 1-3 leave one match out, tested with the given carry; the team
side from the labels as in P9d, so the score isolates the dead ball. CPU.

    python analysis/p9f_dead_context.py <test_carry>      # -> eda_out/reports/P9f_dead_context_<test_carry>.md
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
from mtah.deadball import features_v2, features_v3  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402

TRAIN_P = 0.25
FEATS = {"v2": features_v2, "v3": features_v3}
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
    tr = {s: [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=TRAIN_P), seed=s) for r, k in raws] for s in DRAWS}
    rows = []
    for p, featurize in FEATS.items():
        Dt = [featurize(m) for m in tests]
        Dn = {s: [featurize(m) for m in tr[s]] for s in DRAWS}
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
            r = {"match": m["name"], "feats": p, "spellF1": sc, "dead_share": float(dead[g].mean()), "true_dead_share": float((m["labels"][g] == pm.D).mean())}
            rows.append(r)
            print(r, flush=True)
    df = pd.DataFrame(rows)
    piv = df.pivot(index="match", columns="feats", values="spellF1")
    lines = ["# P9f: ölü topta uzun ufuklu bağlam, test taşıması {} (HistGB, takım etiketten)".format(test_carry), "",
             "`analysis/p9f_dead_context.py {}`.".format(test_carry), "", "| maç | v2 | v3 (bağlamlı) |", "|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} |".format(i, piv.loc[i, "v2"], piv.loc[i, "v3"]) for i in piv.index]
    lines += ["| ortalama | {:.3f} | {:.3f} |".format(piv.v2.mean(), piv.v3.mean()), ""]
    for src in ("DFL", "metrica"):
        q = piv[piv.index.str.startswith(src)]
        lines.append("{}: v3 − v2 {:+.3f}, {} / {} maçta önde.".format(src, (q.v3 - q.v2).mean(), int((q.v3 > q.v2).sum()), len(q)))
    out = os.path.join(ROOT, "eda_out", "reports", "P9f_dead_context_{}.md".format(test_carry))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
