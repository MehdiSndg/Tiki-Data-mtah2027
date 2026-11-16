"""A big wide TCN (P9ao: dilations up to 128, about +-20 s, beat the production TCN; P9at: a bigger GRU won in the full
chain): the TCN alone with dilations 1-128 at 128 hidden (P9ao's) vs 256 hidden, each averaged over 6 seeds (3-8), DFL
leave one match out and Metrica 1-3, team call only, flat switch; also scored against labels merged at 3.0 s. GPU.

    python analysis/p9ax_big_tcn.py [only=<match>,...]      # -> eda_out/reports/P9ax_big_tcn.md
"""

import glob
from functools import partial
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


SEEDS = (3, 4, 5, 6, 7, 8)
FPS = 25
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",") if x]
VARIANTS = ("tcn128", "tcn128h256")
OUT = os.path.join(ROOT, "eda_out", "reports", "P9ax_big_tcn{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


D128 = (1, 2, 4, 8, 16, 32, 64, 128)
NETS = {"tcn128": partial(G._tcn, dilations=D128), "tcn128h256": partial(G._tcn, hidden=256, dilations=D128)}


def fit_predict(train, test, make_net=None, jitter=False):
    allF = np.concatenate([d[0] for d in train])
    mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
    data = [(mod.norm(F), y, w, g) for F, y, w, g in train]
    nets = [G.train(data, s, "cuda", make_net=make_net, jitter=jitter) for s in SEEDS]
    return np.mean([G._predict_one(n, mod.norm(test[0]), test[1], "cuda") for n in nets], axis=0)


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    base = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    feats = [(m, features(m), features(swap_teams(m))) for m in base]
    print("features done", flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    rows = []
    names = [m["name"] for m in base]
    for t in range(len(raws)):
        if ONLY and not any(o in names[t] for o in ONLY):
            continue
        r = {"match": names[t]}
        tr = []
        for k, (m, F, Fs) in enumerate(feats):
            if k == t:
                continue
            sw, sh = swap_teams(m), G.short_mask(m["labels"])
            for mm, FF in ((m, F), (sw, Fs)):
                tr.append((FF, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
        m, F, _ = feats[t]
        g, alive = m["half"] > 0, np.isin(m["labels"], (pm.H, pm.A))
        for v in VARIANTS:
            p = fit_predict(tr, (F, g), make_net=NETS[v])
            pred = np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8)
            r[v] = pm.spell_scores(m["labels"], pred, pm.valid_runs(m["labels"]))["mean_f1"]
            l3 = merged(m["labels"], 3.0)
            r[v + " @3s"] = pm.spell_scores(l3, pred, pm.valid_runs(l3))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P9ax: geniş TCN 128 → 256 gizli (6 tohum, sadece takım)", "", "`analysis/p9ax_big_tcn.py`.", "",
             "| maç | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[v]) for v in VARIANTS) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[v].mean()) for v in VARIANTS) + " |", ""]
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        if len(q):
            for v in VARIANTS[1:]:
                for sfx in ("", " @3s"):
                    lines.append("{}{}: {} − tcn128 {:+.3f}, {} / {}".format(src, sfx, v, (q[v + sfx] - q["tcn128" + sfx]).mean(), int((q[v + sfx] > q["tcn128" + sfx]).sum()), len(q)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
