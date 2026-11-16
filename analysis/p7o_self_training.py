"""Self-training on the test match: the team GRU (with the team swap) retrained with its own confident
predictions on the unlabelled test match as extra targets. DFL LOMO, team call only, flat switch. GPU.

    python analysis/p7o_self_training.py      # -> eda_out/reports/P7o_self_training.md

Algorithmic pseudo-labels on the provided game are allowed (organiser rules, 23 Oct). The point is the
domain gap between our ATD and the degradation model, so each test match is also degraded with shifts
the model was not trained on (P6u). Round 1: trained on the other six (+ swapped). Round 2: + the test
match (and its swap) with target P(home) > 0.5 and weight 1 where max(p, 1 - p) >= CONF, alive frames only.
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
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p6u_shift_robustness import shifts  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

CONF = 0.9
SHIFTS = ("eğitimdeki", "yumuşatmasız top", "top kaybı ×1.5", "parçalı iz ×1.5")
OUT = os.path.join(ROOT, "eda_out", "reports", "P7o_self_training.md")


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [from_dfl(i) for i in ids]
    train = [degrade(r, dict(rates_for(i), smooth_frames=25), seed=0) for r, i in zip(raws, ids)]
    sw = [swap_teams(m) for m in train]
    T, Ts = [features(m) for m in train], [features(m) for m in sw]
    it = lambda m, F: (F, (m["labels"] == pm.H).astype(np.float32), np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32), m["half"] > 0)
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        base = [it(train[k], T[k]) for k in tr] + [it(sw[k], Ts[k]) for k in tr]
        for name in SHIFTS:
            m = degrade(raws[t], shifts(ids[t])[name], seed=0)
            ms = swap_teams(m)
            g = m["half"] > 0
            alive = np.isin(m["labels"], (pm.H, pm.A))
            F, Fs = features(m), features(ms)
            sc = lambda p: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
            p1 = fit_predict(base, (F, g))
            y = (p1 > 0.5).astype(np.float32)
            w = ((np.maximum(p1, 1 - p1) >= CONF) & alive).astype(np.float32)
            pseudo = [(F, y, w, g), (Fs, 1 - y, w, g)]
            p2 = fit_predict(base + pseudo, (F, g))
            rows.append({"match": m["name"], "shift": name, "round1": sc(p1), "round2": sc(p2), "pseudo_share": float(w[alive].mean()),
                         "pseudo_acc": float(((y == (m["labels"] == pm.H)) [w > 0]).mean())})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P7o: test maçında kendi kendine öğrenme (DFL, maç dışarıda bırakarak, sadece takım, 3 tohum)", "", "`analysis/p7o_self_training.py`, CONF = {}.".format(CONF), "",
             "| bozma | 1. tur | 2. tur | fark | önde | sözde etiket payı | isabeti |", "|---|---|---|---|---|---|---|"]
    for name, d in df.groupby("shift", sort=False):
        lines.append("| {} | {:.3f} | {:.3f} | {:+.3f} | {} / {} | {:.0%} | {:.1%} |".format(name, d.round1.mean(), d.round2.mean(), (d.round2 - d.round1).mean(),
                                                                                 int((d.round2 > d.round1).sum()), len(d), d.pseudo_share.mean(), d.pseudo_acc.mean()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
