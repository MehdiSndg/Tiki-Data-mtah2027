"""Boundary-GRU variants (research report item 8) on P9a's out-of-fold team and dead-ball probabilities: the boundary
GRU retrained per test match with the target radius r in RADII frames and, for r = 10, the positive frames weighted
x3; v14's six-net team mean decoded with 12 (1 - b) + 0.5 b on the out-of-fold dead ball. The test match is rebuilt
as in P9a (carried-dead-ball degradation, seed 0); the training data as P9a's boundary GRU (usual degradation, DFL +
Metrica minus the test match, swapped). Full chain, fresh seeds (3, 4, 5). GPU.

    python analysis/p9p_boundary_variants.py <oof_dir> [only=<match>,...]      # -> eda_out/reports/P9p_boundary_variants.md
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
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402
from p9b_oof_sweeps import HI, LO, load  # noqa: E402

SEEDS = (3, 4, 5)
VARIANTS = {"r10 (v14)": (10, 1.0), "r5": (5, 1.0), "r15": (15, 1.0), "r20": (20, 1.0), "r10 pos x3": (10, 3.0)}
ONLY = [x for a in sys.argv[2:] if a.startswith("only=") for x in a[5:].split(",") if x]
OUT = os.path.join(ROOT, "eda_out", "reports", "P9p_boundary_variants{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


def main(d, *rest):
    Z = load(d)
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    old = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    so = [swap_teams(m) for m in old]
    To, Tso = [features(m) for m in old], [features(m) for m in so]
    print("features done", flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    rows = []
    for t, (r0, key) in enumerate(raws):
        name = old[t]["name"]
        if name not in Z or (ONLY and not any(o in name for o in ONLY)):
            continue
        z = Z[name]
        m = degrade(r0, dict(rates_for(key), smooth_frames=25, dead_carry_p=0.25), seed=0)
        g = m["half"] > 0
        Tt = features(m)
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & z["g"]
        p = (z["pg"] + z["pt"] + z["ps"] + z["mg"] + z["mt"] + z["ms"]) / 6
        ks = [k for k in range(len(old)) if k != t]
        r = {"match": name}
        for v, (rad, pw) in VARIANTS.items():
            bt = lambda mm: G.boundary_target(mm["labels"], rad)
            data = [(F, bt(mm), aw(mm) * np.where(bt(mm) > 0, pw, 1.0).astype(np.float32), mm["half"] > 0)
                    for k in ks for mm, F in ((old[k], To[k]), (so[k], Tso[k]))]
            b = fit_predict(data, (Tt, g), seeds=SEEDS)
            cost = HI * (1 - b) + LO * b
            r[v] = pm.spell_scores(z["labels"], np.where(z["g"], decode(p, z["g"] & ~dead, cost), pm.X).astype(np.int8), pm.valid_runs(z["labels"]))["mean_f1"]
        r["oof b"] = pm.spell_scores(z["labels"], np.where(z["g"], decode(p, z["g"] & ~dead, HI * (1 - z["b"]) + LO * z["b"]), pm.X).astype(np.int8),
                                     pm.valid_runs(z["labels"]))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = list(VARIANTS) + ["oof b"]
    lines = ["# P9p: sınır GRU'su varyantları (OOF takım + ölü top, v14 tam zincir, tohum 3–5)", "", "`analysis/p9p_boundary_variants.py`.", "",
             "| maç | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        if len(q):
            lines.append("| ort. {} | ".format(src) + " | ".join("{:.3f}".format(q[c].mean()) for c in cols) + " |")
    lines.append("")
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        for v in list(VARIANTS)[1:]:
            if len(q):
                lines.append("{} {}: r10'a fark {:+.3f}, {} / {}".format(src, v, (q[v] - q["r10 (v14)"]).mean(), int((q[v] > q["r10 (v14)"]).sum()), len(q)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("only=")])
