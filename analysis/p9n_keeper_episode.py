"""Keeper and touch-episode features for the team call (P9n; both teammates found touch episodes useful, +0.005 and
+0.012; the keeper slot was never read by a feature): the team GRU + TCN (v11 recipe: DFL + Metrica, swapped, short
x2) on the production features against the same plus mtah.teammodel.extra_features. DFL leave one match out and
Metrica 1-3 leave one game out, team call only (dead ball from the labels), flat switch, fresh seeds (3, 4, 5). GPU.

    python analysis/p9n_keeper_episode.py [only=<match>,...]      # -> eda_out/reports/P9n_keeper_episode.md
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
from mtah.teammodel import decode, extra_features, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

SEEDS = (3, 4, 5)
FPS = 25
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",") if x]
VARIANTS = ("base", "keeper+episode")
OUT = os.path.join(ROOT, "eda_out", "reports", "P9n_keeper_episode{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    base = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    data = {"base": [], "keeper+episode": []}
    for m in base:
        sw = swap_teams(m)
        F, Fs = features(m), features(sw)
        data["base"].append((m, F, Fs))
        data["keeper+episode"].append((m, np.concatenate([F, extra_features(m)[0]], 1), np.concatenate([Fs, extra_features(sw)[0]], 1)))
    print("features done", data["keeper+episode"][0][1].shape, flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    rows = []
    names = [m["name"] for m in base]
    for t in range(len(raws)):
        if ONLY and not any(o in names[t] for o in ONLY):
            continue
        r = {"match": names[t]}
        for v in VARIANTS:
            tr = []
            for k, (m, F, Fs) in enumerate(data[v]):
                if k == t:
                    continue
                sw, sh = swap_teams(m), G.short_mask(m["labels"])
                for mm, FF in ((m, F), (sw, Fs)):
                    tr.append((FF, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
            m, F, _ = data[v][t]
            g, alive = m["half"] > 0, np.isin(m["labels"], (pm.H, pm.A))
            p = (fit_predict(tr, (F, g), seeds=SEEDS) + fit_predict(tr, (F, g), make_net=G._tcn, seeds=SEEDS)) / 2
            r[v] = pm.spell_scores(m["labels"], np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P9n: kaleci + dokunuş epizotu özellikleri (GRU + TCN, sadece takım, tohum 3–5)", "", "`analysis/p9n_keeper_episode.py`.", "",
             "| maç | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[v]) for v in VARIANTS) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[v].mean()) for v in VARIANTS) + " |", ""]
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        if len(q):
            lines.append("{}: keeper+episode − base {:+.3f}, {} / {}".format(src, (q["keeper+episode"] - q.base).mean(), int((q["keeper+episode"] > q.base).sum()), len(q)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
