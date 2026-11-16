"""More seeds for the dead-ball GRU (P9ad: 6 seeds instead of 3 helped the team nets; P9ae: not the boundary GRU). On
P9a's out-of-fold chains (v14's six-net team, its boundary GRU, the HistGB dead ball qh), the dead-ball GRU is
retrained per test match exactly as in P9a (features_v2 on three carried-dead-ball draws of the other nine matches)
with seeds (6, 7, 8) and averaged with P9a's seeds (3, 4, 5) (qg): s3 = qg, s6 = (qg + new) / 2; dead = to_blocks((qh +
q) / 2). Full chain, DFL and Metrica, also scored against labels merged at 3.0 s. GPU (local RTX 4060).

    python analysis/p9ag_dead_seeds.py <oof_dir> [only=<match>,...]      # -> eda_out/reports/P9ag_dead_seeds.md
"""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402
from p9a_oof import DRAWS  # noqa: E402
from p9b_oof_sweeps import HI, LO, load  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

ONLY = [x for a in sys.argv[2:] if a.startswith("only=") for x in a[5:].split(",") if x]
OUT = os.path.join(ROOT, "eda_out", "reports", "P9ag_dead_seeds{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


def main(d, *rest):
    Z = load(d)
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    new = {s: [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=s) for r, k in raws] for s in DRAWS}
    Dn = {s: [features_v2(m).values.astype(np.float32) for m in new[s]] for s in DRAWS}
    print("features done", flush=True)
    rows = []
    for t in range(len(raws)):
        m = new[0][t]
        name = m["name"]
        if name not in Z or (ONLY and not any(o in name for o in ONLY)):
            continue
        t0 = time.time()
        z = Z[name]
        g = z["g"]
        ks = [k for k in range(len(raws)) if k != t]
        q_new = fit_predict([(Dn[s][k], (new[s][k]["labels"] == pm.D).astype(np.float32), (new[s][k]["half"] > 0).astype(np.float32), new[s][k]["half"] > 0)
                             for s in DRAWS for k in ks], (Dn[0][t], g), seeds=(6, 7, 8))
        p = (z["pg"] + z["pt"] + z["ps"] + z["mg"] + z["mt"] + z["ms"]) / 6
        cost = HI * (1 - z["b"]) + LO * z["b"]
        lab, l3 = z["labels"], merged(z["labels"], 3.0)
        r = {"match": name}
        for v, q in (("s3", z["qg"]), ("s6", (z["qg"] + q_new) / 2)):
            dead = to_blocks((z["qh"] + q) / 2) & g
            pred = np.where(g, decode(p, g & ~dead, cost), pm.X).astype(np.int8)
            r[v] = pm.spell_scores(lab, pred, pm.valid_runs(lab))["mean_f1"]
            r[v + " @3s"] = pm.spell_scores(l3, pred, pm.valid_runs(l3))["mean_f1"]
        rows.append(r)
        print(r, "{:.0f} s".format(time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    cols = ["s3", "s6", "s3 @3s", "s6 @3s"]
    lines = ["# P9ag: ölü top GRU'sunda 3 → 6 tohum (OOF v14 tam zincir)", "", "`analysis/p9ag_dead_seeds.py`.", "",
             "| maç | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
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
    main(*sys.argv[1:])
