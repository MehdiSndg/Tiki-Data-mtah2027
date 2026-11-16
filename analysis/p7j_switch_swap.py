"""Switch cost for the team-swap GRUs, chosen per test match on the other six (nested). Full chain v8 recipe
on the carried-dead-ball DFL. GPU for the predictions, then decode-only sweeps.

    python analysis/p7j_switch_swap.py      # -> eda_out/reports/P7j_switch_swap.md

Per test match: team + boundary GRUs with the team swap (3 seeds), dead ball the v8 mean (HistGB + GRU on
the v2 features, carried degradation). Flat costs FLAT and boundary pairs PAIRS; for each test match the
setting best on the other six is used. Production today: (SWITCH_HI, SWITCH_LO) = (12, 0.5).
"""

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
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

FLAT = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0)
PAIRS = [(hi, lo) for hi in (8.0, 12.0, 16.0, 24.0) for lo in (0.5, 2.0, 4.0)]
OUT = os.path.join(ROOT, "eda_out", "reports", "P7j_switch_swap.md")
CACHE = os.path.join(ROOT, "cache", "p7j_preds.npz")


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
    new = [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=0) for r, k in raws]
    To, Tso, Tn = [features(m) for m in old], [features(m) for m in so], [features(m) for m in new[:nd]]
    Dn = [features_v2(m) for m in new]
    it = lambda m, F, y: (F, y, np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32), m["half"] > 0)
    team = lambda mm: (mm["labels"] == pm.H).astype(np.float32)
    bnd = lambda mm: G.boundary_target(mm["labels"])
    out = {}
    for t in range(nd):
        tr = [k for k in range(nd) if k != t]
        trd = tr + [nd, nd + 1]
        g = new[t]["half"] > 0
        p = fit_predict([it(old[k], To[k], team(old[k])) for k in tr] + [it(so[k], Tso[k], team(so[k])) for k in tr], (Tn[t], g))
        b = fit_predict([it(old[k], To[k], bnd(old[k])) for k in tr] + [it(so[k], Tso[k], bnd(so[k])) for k in tr], (Tn[t], g))
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
            pd.concat([Dn[k][new[k]["half"] > 0] for k in trd], ignore_index=True), np.concatenate([new[k]["labels"][new[k]["half"] > 0] == pm.D for k in trd]))
        qh = np.zeros(len(g))
        qh[g] = clf.predict_proba(Dn[t][g])[:, 1]
        qg = fit_predict([(Dn[k].values.astype(np.float32), (new[k]["labels"] == pm.D).astype(np.float32), (new[k]["half"] > 0).astype(np.float32), new[k]["half"] > 0) for k in trd],
                         (Dn[t].values.astype(np.float32), g))
        out[t] = {"name": new[t]["name"], "labels": new[t]["labels"], "g": g, "p": p, "b": b, "dead": to_blocks((qh + qg) / 2) & g}
        print("preds", t, flush=True)
    np.savez_compressed(CACHE, out=np.array(out, dtype=object))
    return out


def main():
    P = preds()
    n = len(P)
    def sc(t, cost):
        d = P[t]
        return pm.spell_scores(d["labels"], np.where(d["g"], decode(d["p"], d["g"] & ~d["dead"], cost), pm.X).astype(np.int8), pm.valid_runs(d["labels"]))["mean_f1"]
    flat = {(t, s): sc(t, s) for t in range(n) for s in FLAT}
    pair = {(t, pr): sc(t, pr[0] * (1 - P[t]["b"]) + pr[1] * P[t]["b"]) for t in range(n) for pr in PAIRS}
    rows = []
    for t in range(n):
        o = [u for u in range(n) if u != t]
        sf = max(FLAT, key=lambda s: np.mean([flat[(u, s)] for u in o]))
        sp = max(PAIRS, key=lambda pr: np.mean([pair[(u, pr)] for u in o]))
        rows.append({"match": P[t]["name"], "prod": pair[(t, (12.0, 0.5))], "flat": flat[(t, sf)], "flat_s": sf, "pair": pair[(t, sp)], "pair_s": sp})
    df = pd.DataFrame(rows)
    lines = ["# P7j: takım değişimli GRU'lar için değişim cezası (iç içe seçim, tam zincir v8 tarifi, taşınan ölü toplu DFL)", "", "`analysis/p7j_switch_swap.py`.", "",
             "| maç | üretim (12, 0.5) | sabit (seçilen) | sınırlı (seçilen) |", "|---|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} ({:g}) | {:.3f} {} |".format(r["match"], r["prod"], r["flat"], r["flat_s"], r["pair"], r["pair_s"]) for r in rows]
    lines += ["| ortalama | {:.3f} | {:.3f} | {:.3f} |".format(df["prod"].mean(), df["flat"].mean(), df["pair"].mean()), "",
              "Bütün maçlarda ortalama, sabit: " + ", ".join("{:g}: {:.3f}".format(s, np.mean([flat[(t, s)] for t in range(n)])) for s in FLAT),
              "", "Sınırlı: " + ", ".join("{}: {:.3f}".format(pr, np.mean([pair[(t, pr)] for t in range(n)])) for pr in PAIRS)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
