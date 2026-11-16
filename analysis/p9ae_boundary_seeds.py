"""More seeds for the boundary GRU too (P9ad: 6 seeds instead of 3 helped the team nets, v16): on P9a's out-of-fold team
(v14's six-net mean) and dead-ball probabilities, the boundary GRU retrained per test match as in P9p (r = 10) with seeds
(3, 4, 5) and with seeds 3-8, decoded with 12 (1 - b) + 0.5 b. Full chain, DFL and Metrica, also scored against labels
merged at 3.0 s. GPU.

    python analysis/p9ae_boundary_seeds.py <oof_dir> [only=<match>,...]      # -> eda_out/reports/P9ae_boundary_seeds.md
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
from p9l_rescore_calibrated import merged  # noqa: E402

ONLY = [x for a in sys.argv[2:] if a.startswith("only=") for x in a[5:].split(",") if x]
OUT = os.path.join(ROOT, "eda_out", "reports", "P9ae_boundary_seeds{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


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
        data = [(F, G.boundary_target(mm["labels"]), aw(mm), mm["half"] > 0) for k in ks for mm, F in ((old[k], To[k]), (so[k], Tso[k]))]
        b3 = fit_predict(data, (Tt, g), seeds=(3, 4, 5))
        b6 = (b3 + fit_predict(data, (Tt, g), seeds=(6, 7, 8))) / 2
        l3 = merged(z["labels"], 3.0)
        for v, b in (("s3", b3), ("s6", b6)):
            pred = np.where(z["g"], decode(p, z["g"] & ~dead, HI * (1 - b) + LO * b), pm.X).astype(np.int8)
            r[v] = pm.spell_scores(z["labels"], pred, pm.valid_runs(z["labels"]))["mean_f1"]
            r[v + " @3s"] = pm.spell_scores(l3, pred, pm.valid_runs(l3))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["s3", "s6", "s3 @3s", "s6 @3s"]
    lines = ["# P9ae: sınır GRU'sunda 3 → 6 tohum (OOF takım + ölü top, v14 tam zincir)", "", "`analysis/p9ae_boundary_seeds.py`.", "",
             "| maç | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        if len(q):
            lines.append("| ort. {} | ".format(src) + " | ".join("{:.3f}".format(q[c].mean()) for c in cols) + " |")
    lines.append("")
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        for sfx in ("", " @3s"):
            if len(q):
                lines.append("{}{}: s6 − s3 {:+.3f}, {} / {}".format(src, sfx, (q["s6" + sfx] - q["s3" + sfx]).mean(), int((q["s6" + sfx] > q["s3" + sfx]).sum()), len(q)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("only=")])
