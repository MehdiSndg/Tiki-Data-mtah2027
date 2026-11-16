"""A second stage over the first stage's P(home) in time windows (stacking), LOMO on degraded DFL.

    python analysis/p6l_stacking.py      # -> eda_out/reports/P6l_stacking.md

First stage: mtah.teammodel features with touches, HistGB. For each test match the other six are split
into three pairs; the first stage fitted on four predicts the other two, so every training frame has an
out-of-fold P(home). The second stage (HistGB) sees that probability's log-odds, its centred means over
1, 3, 5 and 10 s, its values 1, 2 and 5 s earlier and later, and a few raw features (ball seen, age and
until next sighting, touch team and age), and is trained on the out-of-fold values; at test time the
first stage fitted on all six feeds it. Both stages are decoded with the Viterbi (switch cost chosen
per test match on the training matches' out-of-fold predictions). Team call only (dead ball from the
labels); degradation as P6b (ball smoothed over 25 frames), no extra copies.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.teammodel import decode, fit, team_features  # noqa: E402

FPS = 25
SWITCHES = (4.0, 8.0, 12.0, 16.0, 24.0)
RAW = ("seen", "age", "until", "last_touch_team", "next_touch_team", "touch_age")
OUT = os.path.join(ROOT, "eda_out", "reports", "P6l_stacking.md")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stack_features(p, F, names):
    z = pd.Series(logit(p))
    cols = {"z": z.values}
    for s in (1, 3, 5, 10):
        cols["mean{}".format(s)] = z.rolling(s * FPS, center=True, min_periods=1).mean().values
    for s in (1, 2, 5):
        cols["lag{}".format(s)] = z.shift(s * FPS).bfill().values
        cols["lead{}".format(s)] = z.shift(-s * FPS).ffill().values
    for r in RAW:
        cols[r] = F[:, names.index(r)]
    return np.column_stack(list(cols.values())).astype(np.float32)


def spell(m, c):
    return pm.spell_scores(m["labels"], c, pm.valid_runs(m["labels"]))["mean_f1"]


def best_switch(ms, ps, alives):
    return max(SWITCHES, key=lambda s: np.mean([spell(m, decode(p, a, s)) for m, p, a in zip(ms, ps, alives)]))


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    met = [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2)]
    allm = ms + met
    D = []
    for m in allm:
        F, names = team_features(m, True)
        a = np.isin(m["labels"], (pm.H, pm.A))
        D.append((F, a, (m["labels"] == pm.H).astype(int)))
        print(m["name"], flush=True)
    first = lambda ks: fit([D[k][0][D[k][1]] for k in ks], [D[k][2][D[k][1]] for k in ks])
    rows = []
    for t in range(len(ms)):
        tr = [k for k in range(len(ms)) if k != t]
        pairs = [tr[0:2], tr[2:4], tr[4:6]]
        oof = {}
        for pr in pairs:
            clf = first([k for k in tr if k not in pr] + [7, 8])
            for k in pr:
                oof[k] = clf.predict_proba(D[k][0])[:, 1]
        S = {k: stack_features(oof[k], D[k][0], names) for k in tr}
        from sklearn.ensemble import HistGradientBoostingClassifier
        second = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(
            np.concatenate([S[k][D[k][1]] for k in tr]), np.concatenate([D[k][2][D[k][1]] for k in tr]))
        p2_oof = {k: second.predict_proba(S[k])[:, 1] for k in tr}
        sw1 = best_switch([ms[k] for k in tr], [oof[k] for k in tr], [D[k][1] for k in tr])
        sw2 = best_switch([ms[k] for k in tr], [p2_oof[k] for k in tr], [D[k][1] for k in tr])
        p1 = first(tr + [7, 8]).predict_proba(D[t][0])[:, 1]
        p2 = second.predict_proba(stack_features(p1, D[t][0], names))[:, 1]
        m, a = ms[t], D[t][1]
        r = {"match": m["name"], "sw1": sw1, "sw2": sw2, "first": spell(m, decode(p1, a, sw1)), "stacked": spell(m, decode(p2, a, sw2)),
             "acc1": np.mean((p1[a] >= 0.5) == D[t][2][a]), "acc2": np.mean((p2[a] >= 0.5) == D[t][2][a])}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    d = df.stacked - df["first"]
    lines = ["# P6l: ikinci katman (stacking), DFL maç dışarıda bırakarak", "", "`analysis/p6l_stacking.py`. Sadece takım kararı.", "",
             "| maç | ceza 1 / 2 | birinci katman | iki katman | kare doğruluğu 1 / 2 |", "|---|---|---|---|---|"]
    lines += ["| {} | {:g} / {:g} | {:.3f} | {:.3f} | {:.3f} / {:.3f} |".format(r["match"], r["sw1"], r["sw2"], r["first"], r["stacked"], r["acc1"], r["acc2"]) for r in rows]
    lines += ["", "Ortalama {:.3f} → {:.3f}, fark {:+.3f} ({:+.3f} .. {:+.3f}; {} / {} maçta iki katman önde).".format(
        df["first"].mean(), df.stacked.mean(), d.mean(), d.min(), d.max(), int((d > 0).sum()), len(d))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
