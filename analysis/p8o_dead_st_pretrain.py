"""The dead-ball GRU pre-trained on SoccerTrack's dead ball (status 0; usable matches, carried-dead-ball
degradation) and fine-tuned on the usual data, full chain with v10's team side, on the carried-dead-ball DFL.
Fresh seeds (3, 4, 5). GPU. P8d found ST pre-training helps the team GRU; the dead ball is the larger gap.

    python analysis/p8o_dead_st_pretrain.py      # -> eda_out/reports/P8o_dead_st_pretrain.md

Per test DFL match: team = mean of GRU and TCN on DFL + Metrica 1-3 (swapped, short x2), boundary GRU on the
same data; dead-ball models on the dead-ball v2 features, 6 DFL + 3 Metrica with three carried-dead-ball
degradation draws: HistGB + GRU (v10) against HistGB + ST-pre-trained GRU, HistGB with ST rows + that GRU, and
the dead ball from the labels.
"""

import copy
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
from mtah import grumodel as G  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample, from_soccertrack, soccertrack_ids, soccertrack_usable  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

SEEDS = (3, 4, 5)
DRAWS = (0, 1, 2)
OUT = os.path.join(ROOT, "eda_out", "reports", "P8o_dead_st_pretrain.md")


def pretrained_predict(dd, pre, test, seeds):
    """fit_predict with each seed's net first trained on pre (same normalisation as dd), then on dd."""
    allF = np.concatenate([d[0] for d in dd])
    mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
    norm = lambda data: [(mod.norm(F), y, w, g) for F, y, w, g in data]
    out = []
    for s in seeds:
        warm = G.train(norm(pre), s, "cuda").train()
        net = G.train(norm(dd), s, "cuda", make_net=lambda nf, wn=warm: copy.deepcopy(wn))
        out.append(G._predict_one(net, mod.norm(test[0]), test[1], "cuda"))
    return np.mean(out, axis=0)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    nd = len(ids)
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    so = [swap_teams(m) for m in old]
    new = {s: [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=s) for r, k in raws] for s in DRAWS}
    To, Tso = [features(m) for m in old], [features(m) for m in so]
    Dn = {s: [features_v2(m) for m in new[s]] for s in DRAWS}
    Sh = [G.short_mask(m["labels"]) for m in old]
    st = [m for m in (from_soccertrack(i) for i in soccertrack_ids()) if soccertrack_usable(m)]
    st = [degrade(m, dict(rates_for("st"), smooth_frames=25, dead_carry_p=0.25), seed=s) for s in DRAWS for m in st]
    Dst = [features_v2(m) for m in st]
    pre = [(F.values.astype(np.float32), (m["labels"] == pm.D).astype(np.float32), (m["half"] > 0).astype(np.float32), m["half"] > 0) for F, m in zip(Dst, st)]
    print("features done; ST dead share", np.mean(np.concatenate([m["labels"][m["half"] > 0] == pm.D for m in st])), flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    team = lambda m: (m["labels"] == pm.H).astype(np.float32)
    rows = []
    for t in range(nd):
        m = new[0][t]
        g = m["half"] > 0
        Tt = features(m)
        ks = [k for k in range(len(old)) if k != t]
        wk = lambda k: aw(old[k]) * np.where(Sh[k], 2.0, 1.0).astype(np.float32)
        tt = [(To[k], team(old[k]), wk(k), old[k]["half"] > 0) for k in ks] + [(Tso[k], team(so[k]), wk(k), so[k]["half"] > 0) for k in ks]
        tb = [(To[k], G.boundary_target(old[k]["labels"]), aw(old[k]), old[k]["half"] > 0) for k in ks] + \
             [(Tso[k], G.boundary_target(so[k]["labels"]), aw(so[k]), so[k]["half"] > 0) for k in ks]
        p = (fit_predict(tt, (Tt, g), seeds=SEEDS) + fit_predict(tt, (Tt, g), make_net=G._tcn, seeds=SEEDS)) / 2
        b = fit_predict(tb, (Tt, g), seeds=SEEDS)
        cost = G.SWITCH_HI * (1 - b) + G.SWITCH_LO * b
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
            pd.concat([Dn[s][k][new[s][k]["half"] > 0] for s in DRAWS for k in ks], ignore_index=True),
            np.concatenate([new[s][k]["labels"][new[s][k]["half"] > 0] == pm.D for s in DRAWS for k in ks]))
        qh = np.zeros(len(g))
        qh[g] = clf.predict_proba(Dn[0][t][g])[:, 1]
        clf_st = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
            pd.concat([Dn[s][k][new[s][k]["half"] > 0] for s in DRAWS for k in ks] + [F[m["half"] > 0] for F, m in zip(Dst, st)], ignore_index=True),
            np.concatenate([new[s][k]["labels"][new[s][k]["half"] > 0] == pm.D for s in DRAWS for k in ks] + [m["labels"][m["half"] > 0] == pm.D for m in st]))
        qhs = np.zeros(len(g))
        qhs[g] = clf_st.predict_proba(Dn[0][t][g])[:, 1]
        dd = [(Dn[s][k].values.astype(np.float32), (new[s][k]["labels"] == pm.D).astype(np.float32), (new[s][k]["half"] > 0).astype(np.float32),
               new[s][k]["half"] > 0) for s in DRAWS for k in ks]
        X = Dn[0][t].values.astype(np.float32)
        qg = fit_predict(dd, (X, g), seeds=SEEDS)
        qs = pretrained_predict(dd, pre, (X, g), SEEDS)
        sc = lambda dead: pm.spell_scores(m["labels"], np.where(g, decode(p, g & ~dead, cost), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"], "v10": sc(to_blocks((qh + qg) / 2) & g), "hgb_grust": sc(to_blocks((qh + qs) / 2) & g),
             "hgbst_grust": sc(to_blocks((qhs + qs) / 2) & g), "truth": sc(m["labels"] == pm.D)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["v10", "hgb_grust", "hgbst_grust", "truth"]
    lines = ["# P8o: ölü top GRU'su SoccerTrack'te ön eğitimli (tam zincir, v10 takım tarafı, taşınan ölü toplu DFL, tohum 3–5)", "",
             "`analysis/p8o_dead_st_pretrain.py`.", "",
             "| maç | v10 (HistGB+GRU) | HistGB+GRU(ST) | HistGB(+ST)+GRU(ST) | etiketten |", "|---|---|---|---|---|"]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", ""]
    lines += ["{}: v10'a fark {:+.3f}, {} / 7 maçta önde.".format(c, (df[c] - df.v10).mean(), int((df[c] > df.v10).sum())) for c in cols[1:3]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
