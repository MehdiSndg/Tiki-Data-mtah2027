"""Per-match feature standardisation (each match by its own graded frames' mean and std) against the global
one, under the P6u degradation shifts. DFL LOMO, team call only, flat switch, team swap, 3 seeds. GPU.

    python analysis/p7v_match_norm.py      # -> eda_out/reports/P7v_match_norm.md

Our ATD sits at some distance from the degradation model; standardising each match by itself takes out
shifts of level and scale in the features (fewer players seen, a smoother ball) before the GRU.
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

OUT = os.path.join(ROOT, "eda_out", "reports", "P7v_match_norm.md")


def self_norm(F, g):
    mu, sd = np.nanmean(F[g], axis=0), np.nanstd(F[g], axis=0) + 1e-6
    return np.nan_to_num((F - mu) / sd).astype(np.float32).clip(-10, 10)


def fit(train, per_match):
    if per_match:
        mod = G.GRUEnsemble([], 0.0, 1.0, "cuda")
        data = [(self_norm(F, g), y, w, g) for F, y, w, g in train]
    else:
        allF = np.concatenate([d[0] for d in train])
        mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
        data = [(mod.norm(F), y, w, g) for F, y, w, g in train]
    mod.nets = [G.train(data, s, "cuda") for s in G.SEEDS]
    mod.per_match = per_match
    return mod


def predict(mod, F, g):
    X = self_norm(F, g) if mod.per_match else mod.norm(F)
    return np.mean([G._predict_one(n, X, g, "cuda") for n in mod.nets], axis=0)


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
        data = [it(train[k], T[k]) for k in tr] + [it(sw[k], Ts[k]) for k in tr]
        mods = {"global": fit(data, False), "per_match": fit(data, True)}
        for name, rates in shifts(ids[t]).items():
            m = degrade(raws[t], rates, seed=0)
            g = m["half"] > 0
            F = features(m)
            alive = np.isin(m["labels"], (pm.H, pm.A))
            r = {"match": m["name"], "shift": name}
            for k, mod in mods.items():
                c = np.where(g, decode(predict(mod, F, g), alive, G.SWITCH), pm.X).astype(np.int8)
                r[k] = pm.spell_scores(m["labels"], c, pm.valid_runs(m["labels"]))["mean_f1"]
            rows.append(r)
            print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P7v: maç başına standardizasyon, bozma kaymalarında (DFL, maç dışarıda bırakarak, sadece takım, takım değişimiyle, 3 tohum)", "",
             "`analysis/p7v_match_norm.py`.", "", "| bozma | genel | maç başına | fark | önde |", "|---|---|---|---|---|"]
    for name, d in df.groupby("shift", sort=False):
        lines.append("| {} | {:.3f} | {:.3f} | {:+.3f} | {} / {} |".format(name, d["global"].mean(), d.per_match.mean(), (d.per_match - d["global"]).mean(),
                                                                         int((d.per_match > d["global"]).sum()), len(d)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
