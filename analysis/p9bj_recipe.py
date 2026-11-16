"""The training recipe, unchanged since P6 (30 s windows every 15 s, 8 epochs) although the members grew (BigGRU 256 x 3,
wide TCN): the BigGRU single-head member (v18's) trained with the production recipe vs 60 s windows (stride 30 s) vs 16
epochs; one degradation draw, teams-swapped twins, seeds 3-5; DFL leave one match out and Metrica 1-3, team call only,
flat switch, also scored against labels merged at 3.0 s (P9aw's harness). Unlike more degradation draws, a recipe change
does not lean the training set toward DFL. GPU.

    python analysis/p9bj_recipe.py [only=<match>,...]      # -> eda_out/reports/P9bj_recipe.md
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
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

SEEDS = (3, 4, 5)
FPS = 25
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",") if x]
VARIANTS = {"base": (30, 15, 8), "win60": (60, 30, 8), "ep16": (30, 15, 16)}      # window s, stride s, epochs
OUT = os.path.join(ROOT, "eda_out", "reports", "P9bj_recipe{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


def fit_predict(train, test, recipe):
    win, stride, epochs = recipe
    G.WIN, G.STRIDE, G.EPOCHS = win * FPS, stride * FPS, epochs
    mu, sd = G.feature_stats([d[0] for d in train])
    mod = G.GRUEnsemble([], mu, sd, "cuda")
    data = [(mod.norm(F), y, w, g) for F, y, w, g in train]
    nets = [G.train(data, s, "cuda", make_net=G.NETS["big"]) for s in SEEDS]
    del data
    return np.mean([G._predict_one(n, mod.norm(test[0]), test[1], "cuda") for n in nets], axis=0)


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    ms = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    rows_all = []
    for m in ms:
        sh = G.short_mask(m["labels"])
        rows_all.append([(features(mm), (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0)
                         for mm in (m, swap_teams(m))])
    print("features done", flush=True)
    rows = []
    for t, m in enumerate(ms):
        if ONLY and not any(o in m["name"] for o in ONLY):
            continue
        tr = [x for k in range(len(ms)) if k != t for x in rows_all[k]]
        g, alive = m["half"] > 0, np.isin(m["labels"], (pm.H, pm.A))
        F = rows_all[t][0][0]
        r = {"match": m["name"]}
        for v, rec in VARIANTS.items():
            p = fit_predict(tr, (F, g), rec)
            pred = np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8)
            r[v] = pm.spell_scores(m["labels"], pred, pm.valid_runs(m["labels"]))["mean_f1"]
            l3 = merged(m["labels"], 3.0)
            r[v + " @3s"] = pm.spell_scores(l3, pred, pm.valid_runs(l3))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P9bj: eğitim tarifi (BigGRU, sadece takım, tohum 3–5)", "", "`analysis/p9bj_recipe.py`.", "",
             "| maç | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[v]) for v in VARIANTS) + " |" for r in rows]
    lines.append("")
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        if len(q):
            for v in ("win60", "ep16"):
                for sfx in ("", " @3s"):
                    d = q[v + sfx] - q["base" + sfx]
                    lines.append("{}{}: {} − base {:+.4f}, {} / {}".format(src, sfx, v, d.mean(), int((d > 0).sum()), len(q)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
