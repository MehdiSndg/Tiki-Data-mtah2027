"""Player-level input (mtah.setgru): each team's 11 players nearest the ball as a set, beside the team
features. DFL LOMO, team call only, 3 seeds. GPU.

    python analysis/p7d_player_set.py      # -> eda_out/reports/P7d_player_set.md

Per DFL test match (degraded as production, smooth 25), trained on the other six: mtah.grumodel's GRU on
the production features, and SetNet on the same features + mtah.setgru.player_set. Flat switch, dead ball
from the labels; with the boundary GRU's cost as well (boundary GRU on the production features, shared).
"""

import glob
import os
import sys
from functools import partial

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
from mtah.setgru import player_set, set_net  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P7d_player_set.md")


def fit_predict(train, test, make_net=None, seeds=G.SEEDS):
    allF = np.concatenate([d[0] for d in train])
    mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
    data = [(mod.norm(F), y, w, g) for F, y, w, g in train]
    nets = [G.train(data, s, "cuda", make_net=make_net) for s in seeds]
    return np.mean([G._predict_one(n, mod.norm(test[0]), test[1], "cuda") for n in nets], axis=0)


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    T = [features(m).astype(np.float32) for m in ms]
    TP = [np.concatenate([t, player_set(m)], axis=1) for t, m in zip(T, ms)]
    nft = T[0].shape[1]
    print("features done", nft, TP[0].shape[1], flush=True)
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        m = ms[t]
        g = m["half"] > 0
        lab = lambda k: ms[k]["labels"]
        al = lambda k: np.isin(lab(k), (pm.H, pm.A)).astype(np.float32)
        alive = np.isin(m["labels"], (pm.H, pm.A))
        p0 = fit_predict([(T[k], (lab(k) == pm.H).astype(np.float32), al(k), ms[k]["half"] > 0) for k in tr], (T[t], g))
        p1 = fit_predict([(TP[k], (lab(k) == pm.H).astype(np.float32), al(k), ms[k]["half"] > 0) for k in tr], (TP[t], g), make_net=partial(lambda nf, n0: set_net(n0), n0=nft))
        b = fit_predict([(T[k], G.boundary_target(lab(k)), al(k), ms[k]["half"] > 0) for k in tr], (T[t], g))
        cost = G.SWITCH_HI * (1 - b) + G.SWITCH_LO * b
        sc = lambda p, c: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, c), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"], "gru": sc(p0, G.SWITCH), "set": sc(p1, G.SWITCH), "gru_b": sc(p0, cost), "set_b": sc(p1, cost), "mean_b": sc((p0 + p1) / 2, cost)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["gru", "set", "gru_b", "set_b", "mean_b"]
    lines = ["# P7d: oyuncu kümesi girdisi (DFL, maç dışarıda bırakarak, 3 tohum, sadece takım kararı)", "", "`analysis/p7d_player_set.py`.", "",
             "| maç | GRU | küme GRU | GRU + sınır | küme + sınır | ikisinin ort. + sınır |", "|---|---|---|---|---|---|"]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", "",
              "küme − GRU: {:+.3f} ({} / 7); sınırla {:+.3f} ({} / 7); ortalama sınırla {:+.3f} ({} / 7).".format(
                  (df.set - df.gru).mean(), int((df.set > df.gru).sum()), (df.set_b - df.gru_b).mean(), int((df.set_b > df.gru_b).sum()),
                  (df.mean_b - df.gru_b).mean(), int((df.mean_b > df.gru_b).sum()))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
