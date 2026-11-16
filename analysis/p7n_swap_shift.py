"""Does the team swap's gain (P7e) survive degradations the GRU was not trained on? DFL LOMO, team call only,
flat switch, 3 seeds. GPU.

    python analysis/p7n_swap_shift.py      # -> eda_out/reports/P7n_swap_shift.md

Both GRUs train on the production degradation; one also on the swapped matches. Each test match scored
under the shifts of P6u (p6u_shift_robustness.shifts).
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

OUT = os.path.join(ROOT, "eda_out", "reports", "P7n_swap_shift.md")


def fit(train):
    allF = np.concatenate([d[0] for d in train])
    mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
    mod.nets = [G.train([(mod.norm(F), y, w, g) for F, y, w, g in train], s, "cuda") for s in G.SEEDS]
    return mod


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
        base = fit([it(train[k], T[k]) for k in tr])
        swap = fit([it(train[k], T[k]) for k in tr] + [it(sw[k], Ts[k]) for k in tr])
        for name, rates in shifts(ids[t]).items():
            m = degrade(raws[t], rates, seed=0)
            g = m["half"] > 0
            F = features(m)
            alive = np.isin(m["labels"], (pm.H, pm.A))
            sc = lambda mod: pm.spell_scores(m["labels"], np.where(g, decode(np.mean([G._predict_one(n, mod.norm(F), g, "cuda") for n in mod.nets], axis=0), alive, G.SWITCH), pm.X).astype(np.int8),
                                             pm.valid_runs(m["labels"]))["mean_f1"]
            rows.append({"match": m["name"], "shift": name, "base": sc(base), "swap": sc(swap)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P7n: takım değişiminin kazancı bozma kaymasında (DFL, maç dışarıda bırakarak, sadece takım, 3 tohum)", "", "`analysis/p7n_swap_shift.py`.", "",
             "| bozma | GRU | GRU + takım değişimi | fark | önde |", "|---|---|---|---|---|"]
    for name, d in df.groupby("shift", sort=False):
        lines.append("| {} | {:.3f} | {:.3f} | {:+.3f} | {} / {} |".format(name, d.base.mean(), d.swap.mean(), (d.swap - d.base).mean(), int((d.swap > d.base).sum()), len(d)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
