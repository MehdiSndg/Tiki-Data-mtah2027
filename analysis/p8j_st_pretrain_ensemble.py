"""SoccerTrack pre-training (P8d) for all three members of v11's team ensemble (GRU, TCN, player-set GRU), the
ensemble against v11. DFL LOMO, team call only, flat switch, fresh seeds (3, 4, 5). GPU.

    MTAH_SOCCERTRACK=... python analysis/p8j_st_pretrain_ensemble.py [metrica]      # -> eda_out/reports/P8j_st_pretrain_ensemble[_metrica].md

metrica: the second judge, each Metrica game held out in turn (trained on DFL + the other two games).

Each member: v11 (trained on DFL + Metrica 1-3, swapped, short x2) or first trained on the 8 usable SoccerTrack
matches (swapped, short x2) then fine-tuned on v11's data from those weights. Standardisation from the fine-tuning
data of the fold, for both stages.
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
from mtah.setgru import player_set  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p8d_st_pretrain import usable  # noqa: E402

SEEDS = (3, 4, 5)
METRICA = "metrica" in sys.argv[1:]
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",")]  # rerun some test matches
OUT = os.path.join(ROOT, "eda_out", "reports", "P8j_st_pretrain_ensemble{}.md".format("_metrica" if METRICA else ""))


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    ms += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    st = [degrade(m, dict(rates_for("st"), smooth_frames=25), seed=0) for m in (from_soccertrack(i) for i in soccertrack_ids()) if usable(m)]
    allm = ms + st
    sw = [swap_teams(m) for m in allm]
    T, Ts = [features(m) for m in allm], [features(m) for m in sw]
    P = [np.concatenate([t, player_set(m)], 1) for t, m in zip(T, allm)]
    Ps = [np.concatenate([t, player_set(m)], 1) for t, m in zip(Ts, sw)]
    S = [G.short_mask(m["labels"]) for m in allm]
    nd, nf_ = len(ids), len(ms)
    print("features done", nd, nf_ - nd, len(st), flush=True)
    def data(ks, A, As, norm):
        out = []
        for k in ks:
            for m, F in ((allm[k], A[k]), (sw[k], As[k])):
                w = np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32) * np.where(S[k], 2.0, 1.0).astype(np.float32)
                out.append((norm(F), (m["labels"] == pm.H).astype(np.float32), w, m["half"] > 0))
        return out
    st_ks = list(range(nf_, len(allm)))
    members = {"gru": (T, Ts, G.NETS["gru"]), "tcn": (T, Ts, G.NETS["tcn"]), "set": (P, Ps, G.NETS["set"])}
    rows = []
    for t in (range(nd, nf_) if METRICA else range(nd)):
        ks = [k for k in range(nf_) if k != t]
        m = allm[t]
        if ONLY and not any(o in m["name"] for o in ONLY):
            continue
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        sc = lambda q: pm.spell_scores(m["labels"], np.where(g, decode(q, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        base, tuned = {}, {}
        for name, (A, As, make) in members.items():
            allF = np.concatenate([A[k] for k in ks] + [As[k] for k in ks])
            mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
            fine, pre, X = data(ks, A, As, mod.norm), data(st_ks, A, As, mod.norm), mod.norm(A[t])
            b, u = [], []
            for s in SEEDS:
                b.append(G._predict_one(G.train(fine, s, "cuda", make_net=make), X, g, "cuda"))
                warm = G.train(pre, s, "cuda", make_net=make).train()
                u.append(G._predict_one(G.train(fine, s, "cuda", make_net=lambda nf, w=warm: copy.deepcopy(w)), X, g, "cuda"))
            base[name], tuned[name] = np.mean(b, axis=0), np.mean(u, axis=0)
        r = {"match": m["name"], "v11": sc(np.mean(list(base.values()), axis=0)), "v11_st": sc(np.mean(list(tuned.values()), axis=0))}
        for name in members:
            r[name], r[name + "_st"] = sc(base[name]), sc(tuned[name])
        rows.append(r)
        print({k: (round(v, 3) if not isinstance(v, str) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    cols = ["v11", "v11_st", "gru", "gru_st", "tcn", "tcn_st", "set", "set_st"]
    lines = ["# P8j: SoccerTrack ön eğitimi v11 topluluğunun üç üyesinde ({}, sadece takım, tohum 3–5)".format("Metrica" if METRICA else "DFL"), "", "`analysis/p8j_st_pretrain_ensemble.py`.", "",
             "| maç | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", "",
              "Topluluk: ön eğitimli − v11 {:+.3f}, {} / {} maçta önde.".format((df.v11_st - df.v11).mean(), int((df.v11_st > df.v11).sum()), len(df))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
