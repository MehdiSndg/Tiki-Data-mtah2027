"""v11's three team members (GRU, TCN, player-set GRU) each trained single-task (v11) or multi-task with the boundary
as an auxiliary head (P8m), the ensemble of team heads decoded with v11b's flat switch and with the boundary GRU.
DFL LOMO, team call (dead ball from the labels), fresh seeds (3, 4, 5). GPU.

    python analysis/p8n_multitask_ensemble.py [metrica]      # -> eda_out/reports/P8n_multitask_ensemble[_metrica].md

metrica: the second judge, each Metrica game held out in turn (trained on DFL + the other two games).

Multi-task: the member's net with its output layer widened to two logits (team, boundary), trained with p8m's
mt_train (summed losses; team on alive frames with short possessions x2, boundary on alive frames).
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
from mtah.setgru import player_set  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
import p8m_multitask as MT  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

SEEDS = (3, 4, 5)
FLAT = 4.0
METRICA = "metrica" in sys.argv[1:]
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",")]  # rerun some test matches
OUT = os.path.join(ROOT, "eda_out", "reports", "P8n_multitask_ensemble{}.md".format("_metrica" if METRICA else ""))


def widened(make, nf):
    import torch
    net = make(nf)
    net.out = torch.nn.Linear(net.out.in_features, 2)
    return net


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    ms += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    sw = [swap_teams(m) for m in ms]
    T, Ts = [features(m) for m in ms], [features(m) for m in sw]
    P = [np.concatenate([t, player_set(m)], 1) for t, m in zip(T, ms)]
    Ps = [np.concatenate([t, player_set(m)], 1) for t, m in zip(Ts, sw)]
    S = [G.short_mask(m["labels"]) for m in ms]
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    members = {"gru": (T, Ts, G.NETS["gru"]), "tcn": (T, Ts, G.NETS["tcn"]), "set": (P, Ps, G.NETS["set"])}
    print("features done", flush=True)
    rows = []
    for t in (range(len(ids), len(ms)) if METRICA else range(len(ids))):
        ks = [k for k in range(len(ms)) if k != t]
        m = ms[t]
        if ONLY and not any(o in m["name"] for o in ONLY):
            continue
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        tb = [(F, G.boundary_target(mm["labels"]), aw(mm), mm["half"] > 0) for k in ks for mm, F in ((ms[k], T[k]), (sw[k], Ts[k]))]
        bnd = fit_predict(tb, (T[t], g), seeds=SEEDS)
        single, multi = [], []
        for name, (A, As, make) in members.items():
            pairs = [(mm, F, k) for k in ks for mm, F in ((ms[k], A[k]), (sw[k], As[k]))]
            tt = [(F, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(S[k], 2.0, 1.0).astype(np.float32), mm["half"] > 0) for mm, F, k in pairs]
            single.append(fit_predict(tt, (A[t], g), make_net=make, seeds=SEEDS))
            allF = np.concatenate([F for _, F, _ in pairs])
            mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
            md = [(mod.norm(F), a[1], a[2], G.boundary_target(mm["labels"]), aw(mm), a[3]) for (mm, F, _), a in zip(pairs, tt)]
            MT.mt_net = partial(widened, make)
            multi.append(np.mean([MT.mt_predict(MT.mt_train(md, s, "cuda"), mod.norm(A[t]), g, "cuda")[:, 0] for s in SEEDS], axis=0))
        sc = lambda p, c: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, c), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        cost = G.SWITCH_HI * (1 - bnd) + G.SWITCH_LO * bnd
        ps, pmu = np.mean(single, axis=0), np.mean(multi, axis=0)
        r = {"match": m["name"], "v11": sc(ps, cost), "v11b": sc(ps, FLAT), "mt_bnd": sc(pmu, cost), "mt_flat": sc(pmu, FLAT)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["v11", "v11b", "mt_bnd", "mt_flat"]
    lines = ["# P8n: v11 topluluğu çok görevli üyelerle ({}, ölü top etiketten, tohum 3–5)".format("Metrica" if METRICA else "DFL"), "", "`analysis/p8n_multitask_ensemble.py`.", "",
             "| maç | v11 (sınır) | v11b (sabit 4) | çok görevli, sınır | çok görevli, sabit 4 |", "|---|---|---|---|---|"]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", ""]
    lines += ["{}: v11'e fark {:+.3f}, {} / {} maçta önde.".format(c, (df[c] - df.v11).mean(), int((df[c] > df.v11).sum()), len(df)) for c in ("v11b", "mt_bnd", "mt_flat")]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
