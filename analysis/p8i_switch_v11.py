"""Switch cost for v11's three-member team ensemble, chosen per test match on the other six (nested). DFL LOMO,
team call (dead ball from the labels), fresh seeds (3, 4, 5). GPU for the predictions (cached), then decode
sweeps.

    python analysis/p8i_switch_v11.py      # -> eda_out/reports/P8i_switch_v11.md

Averaging three members softens the probabilities, so the costs set for one GRU (P6s, P7j) may no longer be
best. Production: (SWITCH_HI, SWITCH_LO) = (12, 0.5) with the boundary GRU.
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
FLAT = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
PAIRS = [(hi, lo) for hi in (3.0, 4.0, 6.0, 8.0, 12.0) for lo in (0.25, 0.5, 1.0, 2.0)]
OUT = os.path.join(ROOT, "eda_out", "reports", "P8i_switch_v11.md")
CACHE = os.path.join(ROOT, "cache", "p8i_preds.npz")


def preds():
    if os.path.exists(CACHE):
        with np.load(CACHE, allow_pickle=True) as d:
            return d["out"].item()
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    ms += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    sw = [swap_teams(m) for m in ms]
    T, Ts = [features(m) for m in ms], [features(m) for m in sw]
    P = [np.concatenate([t, player_set(m)], 1) for t, m in zip(T, ms)]
    Ps = [np.concatenate([t, player_set(m)], 1) for t, m in zip(Ts, sw)]
    S = [G.short_mask(m["labels"]) for m in ms]
    mkset = partial(lambda nf, n0: set_net(n0), n0=T[0].shape[1])
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    out = {}
    for t in range(len(ids)):
        ks = [k for k in range(len(ms)) if k != t]
        team = lambda A, As: [(F, (m["labels"] == pm.H).astype(np.float32), aw(m) * np.where(S[k], 2.0, 1.0).astype(np.float32), m["half"] > 0)
                              for k in ks for m, F in ((ms[k], A[k]), (sw[k], As[k]))]
        bnd = [(F, G.boundary_target(m["labels"]), aw(m), m["half"] > 0) for k in ks for m, F in ((ms[k], T[k]), (sw[k], Ts[k]))]
        g = ms[t]["half"] > 0
        p = np.mean([fit_predict(team(T, Ts), (T[t], g), seeds=SEEDS), fit_predict(team(T, Ts), (T[t], g), make_net=G._tcn, seeds=SEEDS),
                     fit_predict(team(P, Ps), (P[t], g), make_net=mkset, seeds=SEEDS)], axis=0)
        out[t] = {"name": ms[t]["name"], "labels": ms[t]["labels"], "g": g, "p": p, "b": fit_predict(bnd, (T[t], g), seeds=SEEDS)}
        print("preds", t, flush=True)
    np.savez_compressed(CACHE, out=np.array(out, dtype=object))
    return out


def main():
    P = preds()
    n = len(P)
    def sc(t, cost):
        d = P[t]
        alive = np.isin(d["labels"], (pm.H, pm.A))
        return pm.spell_scores(d["labels"], np.where(d["g"], decode(d["p"], alive, cost), pm.X).astype(np.int8), pm.valid_runs(d["labels"]))["mean_f1"]
    flat = {(t, s): sc(t, s) for t in range(n) for s in FLAT}
    pair = {(t, pr): sc(t, pr[0] * (1 - P[t]["b"]) + pr[1] * P[t]["b"]) for t in range(n) for pr in PAIRS}
    rows = []
    for t in range(n):
        o = [u for u in range(n) if u != t]
        sf = max(FLAT, key=lambda s: np.mean([flat[(u, s)] for u in o]))
        sp = max(PAIRS, key=lambda pr: np.mean([pair[(u, pr)] for u in o]))
        rows.append({"match": P[t]["name"], "prod": pair[(t, (12.0, 0.5))], "flat": flat[(t, sf)], "flat_s": sf, "pair": pair[(t, sp)], "pair_s": sp})
    df = pd.DataFrame(rows)
    lines = ["# P8i: v11 topluluğu için değişim cezası (iç içe, DFL, sadece takım, tohum 3–5)", "", "`analysis/p8i_switch_v11.py`.", "",
             "| maç | üretim (12, 0.5) | sabit (seçilen) | sınırlı (seçilen) |", "|---|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} ({:g}) | {:.3f} {} |".format(r["match"], r["prod"], r["flat"], r["flat_s"], r["pair"], r["pair_s"]) for r in rows]
    lines += ["| ortalama | {:.3f} | {:.3f} | {:.3f} |".format(df["prod"].mean(), df["flat"].mean(), df["pair"].mean()), "",
              "Sınırlı − üretim: {:+.3f} ({} / 7).".format((df["pair"] - df["prod"]).mean(), int((df["pair"] > df["prod"]).sum())), "",
              "Bütün maçlarda, sabit: " + ", ".join("{:g}: {:.3f}".format(s, np.mean([flat[(t, s)] for t in range(n)])) for s in FLAT), "",
              "Sınırlı: " + ", ".join("{}: {:.3f}".format(pr, np.mean([pair[(t, pr)] for t in range(n)])) for pr in PAIRS)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
