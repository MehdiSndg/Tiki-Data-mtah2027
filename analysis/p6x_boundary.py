"""A boundary GRU (ASRF-style) to lower the switch cost where a change of team is likely. LOMO on DFL. GPU.

    python analysis/p6x_boundary.py      # -> eda_out/reports/P6x_boundary.md

Per DFL test match, trained on the other six: the team GRU (as P6s, 2 seeds) and a boundary GRU on the
same features whose target is "a change of team within +-R frames" on alive frames (2 seeds). Decoded
with a per-frame switch cost hi * (1 - b) + lo * b, b the boundary probability; (hi, lo) chosen per test
match on the other six (as P6q); the flat cost, chosen the same way, alongside. Team call only.
Also the error kinds of P6w for both, short possessions recalled.
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
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p6v_gru_dead import gru_fit_predict  # noqa: E402

R = 10
FLAT = (4.0, 8.0, 12.0, 16.0)
PAIRS = [(hi, lo) for hi in (12.0, 16.0, 24.0, 32.0) for lo in (0.5, 2.0, 4.0)]
OUT = os.path.join(ROOT, "eda_out", "reports", "P6x_boundary.md")


def boundary_target(labels):
    t = np.zeros(len(labels), np.float32)
    ch = np.flatnonzero(np.isin(labels[1:], (pm.H, pm.A)) & np.isin(labels[:-1], (pm.H, pm.A)) & (labels[1:] != labels[:-1])) + 1
    for c in ch:
        t[max(0, c - R):c + R + 1] = 1
    return t


def short_recall(m, c, max_s=5):
    runs = pm.valid_runs(m["labels"])
    gs, ps = pm.possession_spells(m["labels"], runs), pm.possession_spells(c, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gs, ps), len(gs), 0.5)
    short = np.array([(b - a) / 25 < max_s for a, b, _ in gs])
    return flags[short].mean() if short.any() else np.nan


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    T = [features(m) for m in ms]
    P, B = {}, {}
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        g = ms[t]["half"] > 0
        al = lambda k: np.isin(ms[k]["labels"], (pm.H, pm.A)).astype(np.float32)
        P[t] = gru_fit_predict([(T[k], (ms[k]["labels"] == pm.H).astype(np.float32), al(k), ms[k]["half"] > 0) for k in tr], (T[t], g))
        B[t] = gru_fit_predict([(T[k], boundary_target(ms[k]["labels"]), al(k), ms[k]["half"] > 0) for k in tr], (T[t], g))
        print(ms[t]["name"], "boundary mean", round(float(B[t][np.isin(ms[t]["labels"], (pm.H, pm.A))].mean()), 3), flush=True)
    sc = lambda t, cost: pm.spell_scores(ms[t]["labels"], decode(P[t], np.isin(ms[t]["labels"], (pm.H, pm.A)), cost),
                                         pm.valid_runs(ms[t]["labels"]))["mean_f1"]
    flat = {(t, s): sc(t, s) for t in range(len(ids)) for s in FLAT}
    tv = {(t, pr): sc(t, pr[0] * (1 - B[t]) + pr[1] * B[t]) for t in range(len(ids)) for pr in PAIRS}
    rows = []
    for t in range(len(ids)):
        o = [u for u in range(len(ids)) if u != t]
        sf = max(FLAT, key=lambda s: np.mean([flat[(u, s)] for u in o]))
        sp = max(PAIRS, key=lambda pr: np.mean([tv[(u, pr)] for u in o]))
        alive = np.isin(ms[t]["labels"], (pm.H, pm.A))
        rows.append({"match": ms[t]["name"], "flat": flat[(t, sf)], "flat_cost": sf, "boundary": tv[(t, sp)], "pair": sp,
                     "short_flat": short_recall(ms[t], decode(P[t], alive, sf)),
                     "short_boundary": short_recall(ms[t], decode(P[t], alive, sp[0] * (1 - B[t]) + sp[1] * B[t]))})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    d = df.boundary - df.flat
    lines = ["# P6x: sınır GRU'su ile değişen değişim cezası (DFL, maç dışarıda bırakarak)", "", "`analysis/p6x_boundary.py`. Sadece takım kararı.", "",
             "| maç | sabit ceza | spellF1 | sınırlı (hi, lo) | spellF1 | < 5 s bulunma sabit / sınırlı |", "|---|---|---|---|---|---|"]
    lines += ["| {} | {:g} | {:.3f} | {} | {:.3f} | {:.0%} / {:.0%} |".format(r["match"], r["flat_cost"], r["flat"], r["pair"], r["boundary"],
                                                                       r["short_flat"], r["short_boundary"]) for r in rows]
    lines += ["", "Ortalama: sabit {:.3f}, sınırlı {:.3f}, fark {:+.3f} ({:+.3f} .. {:+.3f}; {} / {} maçta sınırlı önde).".format(
        df.flat.mean(), df.boundary.mean(), d.mean(), d.min(), d.max(), int((d > 0).sum()), len(d))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
