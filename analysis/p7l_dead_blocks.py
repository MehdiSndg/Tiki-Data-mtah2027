"""Dead-ball blocking for the v9 dead-ball probability: threshold, median window, shortest block and the
HistGB / GRU weight, chosen per test match on the other six (nested). Full chain, v9 recipe. GPU, then
decode-only sweeps.

    python analysis/p7l_dead_blocks.py      # -> eda_out/reports/P7l_dead_blocks.md

m1_dead_classifier.to_blocks (0.5, 2 s median, gaps and blocks < 2 s) was set for the first classifier.
Per test match (carried-dead-ball DFL): team + boundary GRUs with the team swap, dead-ball HistGB and GRU
on three degradation draws (v9). Grid GRID; production is (0.5, 2 s, 2 s, weight 0.5).
"""

import glob
import itertools
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
from m1_dead_classifier import SEED as DSEED  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

FPS = 25
DRAWS = (0, 1, 2)
GRID = list(itertools.product((0.4, 0.45, 0.5, 0.55, 0.6), (1, 2, 3), (1, 2, 3), (0.3, 0.5, 0.7)))   # thr, median s, min block s, w_hgb
PROD = (0.5, 2, 2, 0.5)
OUT = os.path.join(ROOT, "eda_out", "reports", "P7l_dead_blocks.md")
CACHE = os.path.join(ROOT, "cache", "p7l_preds.npz")


def blocks(p, thr, win_s, min_s):
    w, mb = int(win_s * FPS), int(min_s * FPS)
    dead = pd.Series(p).rolling(w, center=True, min_periods=1).median().values > thr
    e = np.diff(np.r_[0, (~dead).astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if z - a < mb and a > 0 and z < len(dead):
            dead[a:z] = True
    e = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if z - a < mb:
            dead[a:z] = False
    return dead


def preds():
    if os.path.exists(CACHE):
        with np.load(CACHE, allow_pickle=True) as d:
            return d["out"].item()
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2)]
    nd = len(ids)
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws[:nd]]
    so = [swap_teams(m) for m in old]
    new = {s: [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=s) for r, k in raws] for s in DRAWS}
    To, Tso, Tn = [features(m) for m in old], [features(m) for m in so], [features(m) for m in new[0][:nd]]
    Dn = {s: [features_v2(m) for m in new[s]] for s in DRAWS}
    it = lambda m, F, y: (F, y, np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32), m["half"] > 0)
    team = lambda mm: (mm["labels"] == pm.H).astype(np.float32)
    bnd = lambda mm: G.boundary_target(mm["labels"])
    out = {}
    for t in range(nd):
        tr = [k for k in range(nd) if k != t]
        trd = tr + [nd, nd + 1]
        m = new[0][t]
        g = m["half"] > 0
        p = fit_predict([it(old[k], To[k], team(old[k])) for k in tr] + [it(so[k], Tso[k], team(so[k])) for k in tr], (Tn[t], g))
        b = fit_predict([it(old[k], To[k], bnd(old[k])) for k in tr] + [it(so[k], Tso[k], bnd(so[k])) for k in tr], (Tn[t], g))
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
            pd.concat([Dn[s][k][new[s][k]["half"] > 0] for s in DRAWS for k in trd], ignore_index=True),
            np.concatenate([new[s][k]["labels"][new[s][k]["half"] > 0] == pm.D for s in DRAWS for k in trd]))
        qh = np.zeros(len(g))
        qh[g] = clf.predict_proba(Dn[0][t][g])[:, 1]
        qg = fit_predict([(Dn[s][k].values.astype(np.float32), (new[s][k]["labels"] == pm.D).astype(np.float32), (new[s][k]["half"] > 0).astype(np.float32),
                           new[s][k]["half"] > 0) for s in DRAWS for k in trd], (Dn[0][t].values.astype(np.float32), g))
        out[t] = {"name": m["name"], "labels": m["labels"], "g": g, "p": p, "b": b, "qh": qh, "qg": qg}
        print("preds", t, flush=True)
    np.savez_compressed(CACHE, out=np.array(out, dtype=object))
    return out


def main():
    P = preds()
    n = len(P)
    def sc(t, prm):
        d = P[t]
        thr, win, mb, w = prm
        dead = blocks(w * d["qh"] + (1 - w) * d["qg"], thr, win, mb) & d["g"]
        cost = G.SWITCH_HI * (1 - d["b"]) + G.SWITCH_LO * d["b"]
        return pm.spell_scores(d["labels"], np.where(d["g"], decode(d["p"], d["g"] & ~dead, cost), pm.X).astype(np.int8), pm.valid_runs(d["labels"]))["mean_f1"]
    S = {(t, prm): sc(t, prm) for t in range(n) for prm in GRID}
    rows = []
    for t in range(n):
        o = [u for u in range(n) if u != t]
        best = max(GRID, key=lambda prm: np.mean([S[(u, prm)] for u in o]))
        rows.append({"match": P[t]["name"], "prod": S[(t, PROD)], "nested": S[(t, best)], "chosen": best})
    df = pd.DataFrame(rows)
    d = df.nested - df["prod"]
    top = sorted(GRID, key=lambda prm: -np.mean([S[(t, prm)] for t in range(n)]))[:8]
    lines = ["# P7l: ölü top bloklama ayarları (iç içe seçim, tam zincir v9 tarifi, taşınan ölü toplu DFL)", "", "`analysis/p7l_dead_blocks.py`. Ayar: (eşik, medyan s, en kısa blok s, HistGB ağırlığı).", "",
             "| maç | üretim {} | iç içe seçilen | seçilen ayar |".format(PROD), "|---|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} | {} |".format(r["match"], r["prod"], r["nested"], r["chosen"]) for r in rows]
    lines += ["| ortalama | {:.3f} | {:.3f} | |".format(df["prod"].mean(), df.nested.mean()), "",
              "Fark {:+.3f} ({} / 7 maçta önde).".format(d.mean(), int((d > 0).sum())), "",
              "Bütün maçlarda en iyi 8: " + "; ".join("{} {:.3f}".format(prm, np.mean([S[(t, prm)] for t in range(n)])) for prm in top)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
