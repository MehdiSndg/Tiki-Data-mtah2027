"""Dead-ball training without the X frames (external review: Metrica's frames before a half's first event and after its
last are X, 0.7-2% of its halves, and both dead-ball models count them as alive, y = 0). On P9a's out-of-fold chains
(v14's six-net team, its boundary GRU), the HistGB and the dead-ball GRU (seeds 3-5) are retrained per test match as in
P9a (features_v2 on three carried-dead-ball draws of the other nine matches), once as P9a (X frames as alive) and once
with X frames left out of the fit (HistGB rows dropped, GRU weight 0); dead = to_blocks((qh + qg) / 2). Full chain, DFL
and Metrica, also scored against labels merged at 3.0 s. GPU.

    python analysis/p9ar_dead_xmask.py <oof_dir> [only=<match>,...]      # -> eda_out/reports/P9ar_dead_xmask.md
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
from mtah.deadball import features_v2  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402
from p9a_oof import DRAWS  # noqa: E402
from p9b_oof_sweeps import HI, LO, load  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

ONLY = [x for a in sys.argv[2:] if a.startswith("only=") for x in a[5:].split(",") if x]
OUT = os.path.join(ROOT, "eda_out", "reports", "P9ar_dead_xmask{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


def main(d, *rest):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Z = load(d)
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    new = {s: [degrade(r, dict(rates_for(k), smooth_frames=25, dead_carry_p=0.25), seed=s) for r, k in raws] for s in DRAWS}
    Dn = {s: [features_v2(m) for m in new[s]] for s in DRAWS}
    print("features done", flush=True)
    rows = []
    for t in range(len(raws)):
        m = new[0][t]
        name = m["name"]
        if name not in Z or (ONLY and not any(o in name for o in ONLY)):
            continue
        z = Z[name]
        g = z["g"]
        ks = [k for k in range(len(raws)) if k != t]
        p = (z["pg"] + z["pt"] + z["ps"] + z["mg"] + z["mt"] + z["ms"]) / 6
        cost = HI * (1 - z["b"]) + LO * z["b"]
        lab, l3 = z["labels"], merged(z["labels"], 3.0)
        r = {"match": name}
        for v in ("x alive", "x masked"):
            use = lambda mm: (mm["half"] > 0) & ((mm["labels"] != pm.X) if v == "x masked" else True)
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=DSEED).fit(
                pd.concat([Dn[s][k][use(new[s][k])] for s in DRAWS for k in ks], ignore_index=True),
                np.concatenate([new[s][k]["labels"][use(new[s][k])] == pm.D for s in DRAWS for k in ks]))
            qh = np.zeros(len(g))
            qh[g] = clf.predict_proba(Dn[0][t][g])[:, 1]
            qg = fit_predict([(Dn[s][k].values.astype(np.float32), (new[s][k]["labels"] == pm.D).astype(np.float32), use(new[s][k]).astype(np.float32),
                               new[s][k]["half"] > 0) for s in DRAWS for k in ks], (Dn[0][t].values.astype(np.float32), g), seeds=(3, 4, 5))
            dead = to_blocks((qh + qg) / 2) & g
            pred = np.where(g, decode(p, g & ~dead, cost), pm.X).astype(np.int8)
            r[v] = pm.spell_scores(lab, pred, pm.valid_runs(lab))["mean_f1"]
            r[v + " @3s"] = pm.spell_scores(l3, pred, pm.valid_runs(l3))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["x alive", "x masked", "x alive @3s", "x masked @3s"]
    lines = ["# P9ar: ölü top eğitiminde X kareleri dışarıda (OOF v14 tam zincir)", "", "`analysis/p9ar_dead_xmask.py`.", "",
             "| maç | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines.append("")
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        for sfx in ("", " @3s"):
            if len(q):
                lines.append("{}{}: masked − alive {:+.4f}, {} / {}".format(src, sfx, (q["x masked" + sfx] - q["x alive" + sfx]).mean(),
                                                                          int((q["x masked" + sfx] > q["x alive" + sfx]).sum()), len(q)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
