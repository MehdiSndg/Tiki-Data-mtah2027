"""Is the GRU's gain real? Three seeds, its own switch cost, and the full chain. LOMO on DFL. GPU.

    python analysis/p6s_gru_robust.py      # -> eda_out/reports/P6s_gru_robust.md

As P6r (analysis/p6r_gru.py), for GRU seeds 0, 1, 2 and their mean probability. The switch cost for each
test match is the one with the best mean over the other six (as P6q). Full chain: the dead ball from the
dead-ball v2 classifier trained on the other six DFL matches + Metrica 1-2 (as P6j), its blocks, then the
Viterbi; the booster (production features, no copies) alongside.
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
import p6r_gru  # noqa: E402
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import SEED as DSEED, to_blocks  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.teammodel import decode, fit, team_features  # noqa: E402

SWITCHES = (4.0, 8.0, 12.0, 16.0, 24.0)
SEEDS = (0, 1, 2)
OUT = os.path.join(ROOT, "eda_out", "reports", "P6s_gru_robust.md")


def spell(m, c):
    return pm.spell_scores(m["labels"], c, pm.valid_runs(m["labels"]))["mean_f1"]


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    met = [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2)]
    D, DD = [], []
    for m in ms + met:
        F, _ = team_features(m, True, True)
        D.append((F, np.isin(m["labels"], (pm.H, pm.A)), (m["labels"] == pm.H).astype(np.float32), m["half"] > 0))
        DD.append((features_v2(m), m["half"] > 0, m["labels"] == pm.D))
        print(m["name"], flush=True)
    # per test match: probabilities of each model, the alive mask from labels and from the classifier
    probs, alive_true, alive_clf = {}, {}, {}
    for t in range(len(ms)):
        tr = [k for k in range(len(ms)) if k != t]
        allF = np.concatenate([D[k][0] for k in tr])
        mu, sd = np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6
        norm = lambda F: np.nan_to_num((F - mu) / sd).astype(np.float32).clip(-10, 10)
        train = [w for k in tr for w in p6r_gru.windows(norm(D[k][0]), D[k][2], D[k][1].astype(np.float32), D[k][3])]
        F, alive, _, g = D[t]
        P = {}
        for s in SEEDS:
            p6r_gru.SEED = s
            P["gru{}".format(s)] = p6r_gru.predict(p6r_gru.train_gru(train, allF.shape[1], "cuda"), norm(F), g, "cuda")
        P["gru_mean"] = np.mean([P["gru{}".format(s)] for s in SEEDS], axis=0)
        P["booster"] = fit([D[k][0][D[k][1]] for k in tr], [D[k][2][D[k][1]].astype(int) for k in tr]).predict_proba(F)[:, 1]
        trd = tr + [len(ms), len(ms) + 1]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
            pd.concat([DD[k][0][DD[k][1]] for k in trd], ignore_index=True), np.concatenate([DD[k][2][DD[k][1]] for k in trd]))
        q = np.zeros(len(g))
        q[g] = clf.predict_proba(DD[t][0][g])[:, 1]
        probs[t], alive_true[t], alive_clf[t] = P, alive, g & ~(to_blocks(q) & g)
        print(ms[t]["name"], "done", flush=True)
    names = list(probs[0].keys())
    rows = []
    for name in names:
        for kind, AL in (("takım", alive_true), ("tam zincir", alive_clf)):
            sc = {(t, s): spell(ms[t], np.where(ms[t]["half"] > 0, decode(probs[t][name], AL[t], s), pm.X).astype(np.int8))
                  for t in range(len(ms)) for s in SWITCHES}
            nested = []
            for t in range(len(ms)):
                s = max(SWITCHES, key=lambda x: np.mean([sc[(u, x)] for u in range(len(ms)) if u != t]))
                nested.append(sc[(t, s)])
            best_fixed = max(SWITCHES, key=lambda x: np.mean([sc[(u, x)] for u in range(len(ms))]))
            rows.append((name, kind, np.mean(nested), np.min(nested), best_fixed))
            print(rows[-1], flush=True)
    lines = ["# P6s: GRU'nun kazancı sağlam mı? (DFL, maç dışarıda bırakarak)", "", "`analysis/p6s_gru_robust.py`. Değişim cezası iç seçimle.", "",
             "| model | ölçüm | ortalama spellF1 | en kötü maç | en iyi sabit ceza |", "|---|---|---|---|---|"]
    lines += ["| {} | {} | {:.3f} | {:.3f} | {:g} |".format(*r) for r in rows]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
