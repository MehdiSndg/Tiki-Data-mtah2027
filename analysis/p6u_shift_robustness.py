"""Does the GRU's lead survive a degradation it was not trained on? LOMO on DFL. GPU.

    python analysis/p6u_shift_robustness.py      # -> eda_out/reports/P6u_shift_robustness.md

Both models train on the production degradation (ball smoothed over 25 frames, seed 0): the GRU
(mtah.grumodel, 2 seeds averaged) and the booster (mtah.teammodel, production features, no copies).
Each test match is scored (team call only, dead ball from the labels) under the training degradation
and under shifted ones: ball losses x1.5, jitter and drift x1.5, unsmoothed ball, ball smoothed over 50
frames, track fragments x1.5 -- a stand-in for how far our ATD may sit from the degradation model.
Also the mean of the two models' probabilities.
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
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.teammodel import SWITCH as BSWITCH, decode, features, fit  # noqa: E402

SEEDS = (0, 1)
OUT = os.path.join(ROOT, "eda_out", "reports", "P6u_shift_robustness.md")


def shifts(key):
    r = dict(rates_for(key), smooth_frames=25)
    out = {"eğitimdeki": r}
    out["top kaybı ×1.5"] = dict(r, loss_per_min=r["loss_per_min"] * 1.5)
    out["titreme+kayma ×1.5"] = dict(r, jitter_share=r["jitter_share"] * 1.5, drift_p=r["drift_p"] * 1.5)
    out["yumuşatmasız top"] = dict(r, smooth_frames=0)
    out["50 kare yumuşatma"] = dict(r, smooth_frames=50)
    out["parçalı iz ×1.5"] = dict(r, piece_drop=min(1.0, r["piece_drop"] * 1.5))
    return out


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [from_dfl(i) for i in ids]
    train = [degrade(r, dict(rates_for(i), smooth_frames=25), seed=0) for r, i in zip(raws, ids)]
    TF = [features(m) for m in train]
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        allF = np.concatenate([TF[k] for k in tr])
        mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
        data = [(mod.norm(TF[k]), (train[k]["labels"] == pm.H).astype(np.float32),
                 np.isin(train[k]["labels"], (pm.H, pm.A)).astype(np.float32), train[k]["half"] > 0) for k in tr]
        mod.nets = [G.train(data, s, "cuda") for s in SEEDS]
        bst = fit([TF[k][np.isin(train[k]["labels"], (pm.H, pm.A))] for k in tr],
                  [(train[k]["labels"][np.isin(train[k]["labels"], (pm.H, pm.A))] == pm.H).astype(int) for k in tr])
        for name, rates in shifts(ids[t]).items():
            m = degrade(raws[t], rates, seed=7)
            F = features(m)
            g, alive = m["half"] > 0, np.isin(m["labels"], (pm.H, pm.A))
            pg = np.mean([G._predict_one(n, mod.norm(F), g, "cuda") for n in mod.nets], axis=0)
            pb = bst.predict_proba(F)[:, 1]
            sc = lambda c: pm.spell_scores(m["labels"], c, pm.valid_runs(m["labels"]))["mean_f1"]
            rows.append({"match": m["name"], "shift": name, "gru": sc(decode(pg, alive, G.SWITCH)), "booster": sc(decode(pb, alive, BSWITCH)),
                         "mean": sc(decode((pg + pb) / 2, alive, G.SWITCH))})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows).groupby("shift", sort=False)[["booster", "gru", "mean"]].mean()
    lines = ["# P6u: GRU'nun önü bozma değişince kalıyor mu? (DFL, maç dışarıda bırakarak)", "",
             "`analysis/p6u_shift_robustness.py`. Eğitim varsayılan bozmayla; test farklı bozmalarla (tohum 7). Sadece takım kararı.", "",
             "| test bozması | booster | GRU | ortalama | GRU − booster |", "|---|---|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} | {:.3f} | {:+.3f} |".format(k, r.booster, r.gru, r["mean"], r.gru - r.booster) for k, r in df.iterrows()]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
