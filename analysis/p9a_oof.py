"""Out-of-fold probabilities of every v11 part, saved per test match, so decode and dead-ball post-processing can be
studied on the CPU without retraining (research report item 1). P8k's chain (carried-dead-ball DFL leave one match out,
or Metrica 1-3 leave one game out; fresh seeds 3, 4, 5) plus P8n's multi-task members (team + boundary heads). GPU.

    MTAH_OOF_DIR=... python analysis/p9a_oof.py [dfl|metrica] [only=<match>,...]      # -> <MTAH_OOF_DIR>/oof_<match>.npz

Saved per match: labels, g (played), name; team members pg (GRU), pt (TCN), ps (player-set GRU); boundary GRU b;
multi-task team heads mg, mt, ms and their boundary heads mbg, mbt, mbs; dead ball qh (HistGB), qg (GRU).

Team = mean of GRU, TCN and player-set GRU on DFL + Metrica 1-3 minus the test match (swapped, short x2);
boundary GRU on the same data; dead ball v9/v10/v11 (HistGB + GRU on the dead-ball v2 features, three
carried-dead-ball draws, DFL + Metrica minus the test match). Also v10's team side (GRU + TCN) for comparison.
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
from m1_dead_classifier import SEED as DSEED, to_blocks  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.setgru import player_set, set_net  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
import p8m_multitask as MT  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402
from p8n_multitask_ensemble import widened  # noqa: E402

SEEDS = (3, 4, 5)
DRAWS = (0, 1, 2)
OUT_DIR = os.environ.get("MTAH_OOF_DIR", os.path.join(ROOT, "cache", "oof"))
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",")]


def multitask(A, As, make, ks, old, so, Sh, aw, X, g):
    """P8n's two-head member: team head and boundary head on the test match, mean over SEEDS."""
    md0 = [(F, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(Sh[k], 2.0, 1.0).astype(np.float32), G.boundary_target(mm["labels"]),
            aw(mm), mm["half"] > 0) for k in ks for mm, F in ((old[k], A[k]), (so[k], As[k]))]
    allF = np.concatenate([d[0] for d in md0])
    mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
    md = [(mod.norm(d[0]),) + d[1:] for d in md0]
    MT.mt_net = partial(widened, make)
    p = np.mean([MT.mt_predict(MT.mt_train(md, s, "cuda"), mod.norm(X), g, "cuda") for s in SEEDS], axis=0)
    return p[:, 0], p[:, 1]


def main(source="dfl"):
    from sklearn.ensemble import HistGradientBoostingClassifier
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    nd = len(ids)
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    so = [swap_teams(m) for m in old]
    new = {s: [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=s) for r, k in raws] for s in DRAWS}
    To, Tso = [features(m) for m in old], [features(m) for m in so]
    Po = [np.concatenate([t, player_set(m)], 1) for t, m in zip(To, old)]
    Pso = [np.concatenate([t, player_set(m)], 1) for t, m in zip(Tso, so)]
    Dn = {s: [features_v2(m) for m in new[s]] for s in DRAWS}
    Sh = [G.short_mask(m["labels"]) for m in old]
    mkset = partial(lambda nf, n0: set_net(n0), n0=To[0].shape[1])
    print("features done", flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    tests = range(nd) if source == "dfl" else range(nd, nd + 3)
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for t in tests:
        m = new[0][t]
        if ONLY and not any(o in m["name"] for o in ONLY):
            continue
        g = m["half"] > 0
        Tt = features(m)
        Pt = np.concatenate([Tt, player_set(m)], 1)
        ks = [k for k in range(len(old)) if k != t]
        team = lambda A, As: [(F, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(Sh[k], 2.0, 1.0).astype(np.float32), mm["half"] > 0)
                              for k in ks for mm, F in ((old[k], A[k]), (so[k], As[k]))]
        bnd = [(F, G.boundary_target(mm["labels"]), aw(mm), mm["half"] > 0) for k in ks for mm, F in ((old[k], To[k]), (so[k], Tso[k]))]
        pg, pt = fit_predict(team(To, Tso), (Tt, g), seeds=SEEDS), fit_predict(team(To, Tso), (Tt, g), make_net=G._tcn, seeds=SEEDS)
        ps = fit_predict(team(Po, Pso), (Pt, g), make_net=mkset, seeds=SEEDS)
        b = fit_predict(bnd, (Tt, g), seeds=SEEDS)
        cost = G.SWITCH_HI * (1 - b) + G.SWITCH_LO * b
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
            pd.concat([Dn[s][k][new[s][k]["half"] > 0] for s in DRAWS for k in ks], ignore_index=True),
            np.concatenate([new[s][k]["labels"][new[s][k]["half"] > 0] == pm.D for s in DRAWS for k in ks]))
        qh = np.zeros(len(g))
        qh[g] = clf.predict_proba(Dn[0][t][g])[:, 1]
        qg = fit_predict([(Dn[s][k].values.astype(np.float32), (new[s][k]["labels"] == pm.D).astype(np.float32), (new[s][k]["half"] > 0).astype(np.float32),
                           new[s][k]["half"] > 0) for s in DRAWS for k in ks], (Dn[0][t].values.astype(np.float32), g), seeds=SEEDS)
        mg, mbg = multitask(To, Tso, G.NETS["gru"], ks, old, so, Sh, aw, Tt, g)
        mtc, mbt = multitask(To, Tso, G.NETS["tcn"], ks, old, so, Sh, aw, Tt, g)
        msv, mbs = multitask(Po, Pso, mkset, ks, old, so, Sh, aw, Pt, g)
        np.savez_compressed(os.path.join(OUT_DIR, "oof_{}.npz".format(m["name"].split("+")[0])), labels=m["labels"], g=g, name=m["name"],
                            pg=pg, pt=pt, ps=ps, b=b, mg=mg, mt=mtc, ms=msv, mbg=mbg, mbt=mbt, mbs=mbs, qh=qh, qg=qg)
        dead = to_blocks((qh + qg) / 2) & g
        sc = lambda p, dd: pm.spell_scores(m["labels"], np.where(g, decode(p, g & ~dd, cost), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"], "v10": sc((pg + pt) / 2, dead), "v11": sc((pg + pt + ps) / 3, dead), "v11_dead_truth": sc((pg + pt + ps) / 3, m["labels"] == pm.D),
             "mt": sc((mg + mtc + msv) / 3, dead)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["v10", "v11", "v11_dead_truth", "mt"]
    out = os.path.join(ROOT, "eda_out", "reports", "P9a_oof_{}.md".format(source))
    lines = ["# P9a: v11 tam zincir + çok görevli üyeler, OOF dökümü ({}, tohum 3–5)".format(source), "", "`analysis/p9a_oof.py {}`.".format(source), "",
             "| maç | v10 | v11 | v11, ölü top etiketten | çok görevli |", "|---|---|---|---|---|"]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", "",
              "v11 − v10: {:+.3f}, {} / {} maçta önde.".format((df.v11 - df.v10).mean(), int((df.v11 > df.v10).sum()), len(df))]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("only=")])
