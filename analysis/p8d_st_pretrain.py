"""SoccerTrack v2 as pre-training only: the team GRU first trained on the 8 usable SoccerTrack matches, then
fine-tuned on v10's data (DFL + Metrica 1-3, swapped, short x2). DFL LOMO, team call only, flat switch,
fresh seeds (3, 4, 5). GPU.

    MTAH_SOCCERTRACK=... python analysis/p8d_st_pretrain.py      # -> eda_out/reports/P8d_st_pretrain.md

P7z: training on SoccerTrack and DFL together hurts DFL (its possession definition differs) though its
labels are learnable. Pre-training lets the net learn football movement from it and the definition from
DFL. Features standardised on the fine-tuning data of each fold (the same statistics for both stages).
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
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample, from_soccertrack, soccertrack_ids  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402

SEEDS = (3, 4, 5)
OUT = os.path.join(ROOT, "eda_out", "reports", "P8d_st_pretrain.md")


def usable(m):
    g = m["half"] > 0
    return np.mean(np.isin(m["labels"][g], (pm.H, pm.A))) >= 0.3 and all(np.isfinite(m["px"][g][:, m["slot_team"] == t]).any() for t in (0, 1))


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    ms += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    st = [degrade(m, dict(rates_for("st"), smooth_frames=25), seed=0) for m in (from_soccertrack(i) for i in soccertrack_ids()) if usable(m)]
    allm = ms + st
    sw = [swap_teams(m) for m in allm]
    T, Ts = [features(m) for m in allm], [features(m) for m in sw]
    S = [G.short_mask(m["labels"]) for m in allm]
    nd, nf_ = len(ids), len(ms)
    print("features done", nd, nf_ - nd, len(st), flush=True)
    def data(ks, norm):
        out = []
        for k in ks:
            for m, F in ((allm[k], T[k]), (sw[k], Ts[k])):
                w = np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32) * np.where(S[k], 2.0, 1.0).astype(np.float32)
                out.append((norm(F), (m["labels"] == pm.H).astype(np.float32), w, m["half"] > 0))
        return out
    st_ks = list(range(nf_, len(allm)))
    rows = []
    for t in range(nd):
        ks = [k for k in range(nf_) if k != t]
        allF = np.concatenate([T[k] for k in ks] + [Ts[k] for k in ks])
        mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
        fine, pre = data(ks, mod.norm), data(st_ks, mod.norm)
        m = allm[t]
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        X = mod.norm(T[t])
        sc = lambda q: pm.spell_scores(m["labels"], np.where(g, decode(q, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        base, tuned = [], []
        for s in SEEDS:
            base.append(G._predict_one(G.train(fine, s, "cuda"), X, g, "cuda"))
            warm = G.train(pre, s, "cuda").train()
            tuned.append(G._predict_one(G.train(fine, s, "cuda", make_net=lambda nf, w=warm: copy.deepcopy(w)), X, g, "cuda"))
        r = {"match": m["name"], "v10_gru": sc(np.mean(base, axis=0)), "st_pretrained": sc(np.mean(tuned, axis=0))}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    d = df.st_pretrained - df.v10_gru
    lines = ["# P8d: SoccerTrack ile ön eğitim, v10 verisiyle ince ayar (DFL, sadece takım, GRU, tohum 3–5)", "", "`analysis/p8d_st_pretrain.py`.", "",
             "| maç | v10 GRU | SoccerTrack ön eğitimli |", "|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} |".format(r["match"], r["v10_gru"], r["st_pretrained"]) for r in rows]
    lines += ["| ortalama | {:.3f} | {:.3f} |".format(df.v10_gru.mean(), df.st_pretrained.mean()), "",
              "Fark {:+.3f} ({:+.3f} .. {:+.3f}), {} / 7 maçta önde.".format(d.mean(), d.min(), d.max(), int((d > 0).sum()))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
