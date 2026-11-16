"""v11's team ensemble (GRU + TCN + player-set GRU) with each member trained on its own degradation draw, and
log-odds against probability averaging. DFL LOMO, team call only, flat switch, fresh seeds (3, 4, 5). GPU.

    python analysis/p8h_ensemble_diversity.py      # -> eda_out/reports/P8h_ensemble_diversity.md

v11: all three members on the same draw (seed 0) of every training match. diverse: GRU on draw 0, TCN on
draw 1, set on draw 2 (the test match stays on draw 0). Both averaged as probabilities and as log-odds.
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
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.setgru import player_set, set_net  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

SEEDS = (3, 4, 5)
OUT = os.path.join(ROOT, "eda_out", "reports", "P8h_ensemble_diversity.md")


def logit_mean(ps):
    z = np.mean([np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1)) for p in ps], axis=0)
    return 1 / (1 + np.exp(-z))


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    draws = {}
    for d in (0, 1, 2):
        ms = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=d) for r, k in raws]
        sw = [swap_teams(m) for m in ms]
        T, Ts = [features(m) for m in ms], [features(m) for m in sw]
        P = [np.concatenate([t, player_set(m)], 1) for t, m in zip(T, ms)]
        Ps = [np.concatenate([t, player_set(m)], 1) for t, m in zip(Ts, sw)]
        draws[d] = (ms, sw, T, Ts, P, Ps)
        print("draw", d, "features done", flush=True)
    nd, nft = len(ids), draws[0][2][0].shape[1]
    mkset = partial(lambda nf, n0: set_net(n0), n0=nft)
    def data(d, ks, use_set):
        ms, sw, T, Ts, P, Ps = draws[d]
        out = []
        for k in ks:
            sh = G.short_mask(ms[k]["labels"])
            for m, F in ((ms[k], P[k] if use_set else T[k]), (sw[k], Ps[k] if use_set else Ts[k])):
                w = np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32) * np.where(sh, 2.0, 1.0).astype(np.float32)
                out.append((F, (m["labels"] == pm.H).astype(np.float32), w, m["half"] > 0))
        return out
    rows = []
    for t in range(nd):
        ks = [k for k in range(len(raws)) if k != t]
        ms0, _, T0, _, P0, _ = draws[0]
        m = ms0[t]
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        same = [fit_predict(data(0, ks, False), (T0[t], g), seeds=SEEDS), fit_predict(data(0, ks, False), (T0[t], g), make_net=G._tcn, seeds=SEEDS),
                fit_predict(data(0, ks, True), (P0[t], g), make_net=mkset, seeds=SEEDS)]
        div = [same[0], fit_predict(data(1, ks, False), (T0[t], g), make_net=G._tcn, seeds=SEEDS),
               fit_predict(data(2, ks, True), (P0[t], g), make_net=mkset, seeds=SEEDS)]
        sc = lambda q: pm.spell_scores(m["labels"], np.where(g, decode(q, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"], "v11": sc(np.mean(same, axis=0)), "v11_logit": sc(logit_mean(same)),
             "diverse": sc(np.mean(div, axis=0)), "diverse_logit": sc(logit_mean(div))}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["v11", "v11_logit", "diverse", "diverse_logit"]
    lines = ["# P8h: topluluk çeşitliliği (üyeler farklı bozma çekilişleriyle) ve log-odds ortalaması (DFL, sadece takım, tohum 3–5)", "",
             "`analysis/p8h_ensemble_diversity.py`.", "", "| maç | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", ""]
    lines += ["{}: v11'e fark {:+.3f}, {} / 7 maçta önde.".format(c, (df[c] - df.v11).mean(), int((df[c] > df.v11).sum())) for c in cols[1:]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
