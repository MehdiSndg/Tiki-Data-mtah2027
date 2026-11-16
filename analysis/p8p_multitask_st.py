"""P8n's multi-task members (team + boundary heads) with SoccerTrack pre-training (P8j): each member's two-head net
first trained on the usable SoccerTrack matches (swapped, short x2; boundary on alive frames), then fine-tuned on
v11's data from those weights. Against the multi-task ensemble without pre-training; both decoded with the boundary
GRU's per-frame switch cost. Team call (dead ball from the labels), fresh seeds (3, 4, 5). GPU.

    MTAH_SOCCERTRACK=... python analysis/p8p_multitask_st.py [metrica]      # -> eda_out/reports/P8p_multitask_st[_metrica].md

metrica: the second judge, each Metrica game held out in turn (trained on DFL + the other two games).
"""

import copy
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
from mtah.external import DFL, from_dfl, from_metrica_sample, from_soccertrack, soccertrack_ids, soccertrack_usable  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.setgru import player_set  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
import p8m_multitask as MT  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

SEEDS = (3, 4, 5)
FLAT = 4.0
METRICA = "metrica" in sys.argv[1:]
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",")]  # rerun some test matches
OUT = os.path.join(ROOT, "eda_out", "reports", "P8p_multitask_st{}.md".format("_metrica" if METRICA else ""))


def widened(make, nf):
    import torch
    net = make(nf)
    net.out = torch.nn.Linear(net.out.in_features, 2)
    return net


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    ms += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    nf_ = len(ms)
    ms += [degrade(m, dict(rates_for("st"), smooth_frames=25), seed=0) for m in (from_soccertrack(i) for i in soccertrack_ids()) if soccertrack_usable(m)]
    sw = [swap_teams(m) for m in ms]
    T, Ts = [features(m) for m in ms], [features(m) for m in sw]
    P = [np.concatenate([t, player_set(m)], 1) for t, m in zip(T, ms)]
    Ps = [np.concatenate([t, player_set(m)], 1) for t, m in zip(Ts, sw)]
    S = [G.short_mask(m["labels"]) for m in ms]
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    members = {"gru": (T, Ts, G.NETS["gru"]), "tcn": (T, Ts, G.NETS["tcn"]), "set": (P, Ps, G.NETS["set"])}
    print("features done", flush=True)
    rows = []
    st_ks = list(range(nf_, len(ms)))
    for t in (range(len(ids), nf_) if METRICA else range(len(ids))):
        ks = [k for k in range(nf_) if k != t]
        m = ms[t]
        if ONLY and not any(o in m["name"] for o in ONLY):
            continue
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        tb = [(F, G.boundary_target(mm["labels"]), aw(mm), mm["half"] > 0) for k in ks for mm, F in ((ms[k], T[k]), (sw[k], Ts[k]))]
        bnd = fit_predict(tb, (T[t], g), seeds=SEEDS)
        multi, multi_st = [], []
        for name, (A, As, make) in members.items():
            def md_of(kk, norm):
                return [(norm(F), (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(S[k], 2.0, 1.0).astype(np.float32),
                         G.boundary_target(mm["labels"]), aw(mm), mm["half"] > 0) for k in kk for mm, F in ((ms[k], A[k]), (sw[k], As[k]))]
            allF = np.concatenate([A[k] for k in ks] + [As[k] for k in ks])
            mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
            md, pre, X = md_of(ks, mod.norm), md_of(st_ks, mod.norm), mod.norm(A[t])
            a, b = [], []
            for s_ in SEEDS:
                MT.mt_net = partial(widened, make)
                a.append(MT.mt_predict(MT.mt_train(md, s_, "cuda"), X, g, "cuda")[:, 0])
                warm = MT.mt_train(pre, s_, "cuda").train()
                MT.mt_net = lambda nf, w=warm: copy.deepcopy(w)
                b.append(MT.mt_predict(MT.mt_train(md, s_, "cuda"), X, g, "cuda")[:, 0])
            multi.append(np.mean(a, axis=0))
            multi_st.append(np.mean(b, axis=0))
        sc = lambda p, c: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, c), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        cost = G.SWITCH_HI * (1 - bnd) + G.SWITCH_LO * bnd
        pmu, pst = np.mean(multi, axis=0), np.mean(multi_st, axis=0)
        r = {"match": m["name"], "mt_bnd": sc(pmu, cost), "mt_st_bnd": sc(pst, cost)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["mt_bnd", "mt_st_bnd"]
    lines = ["# P8p: çok görevli üyeler SoccerTrack ön eğitimli ({}, ölü top etiketten, tohum 3–5)".format("Metrica" if METRICA else "DFL"), "",
             "`analysis/p8p_multitask_st.py`.", "", "| maç | çok görevli | çok görevli + ST ön eğitimi |", "|---|---|---|"]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", "",
              "mt_st_bnd: mt_bnd'ye fark {:+.3f}, {} / {} maçta önde.".format((df.mt_st_bnd - df.mt_bnd).mean(), int((df.mt_st_bnd > df.mt_bnd).sum()), len(df))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
