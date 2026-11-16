"""More augmentation on top of the team swap (P7e). DFL LOMO, team call only, flat switch. GPU.

    python analysis/p7g_more_aug.py      # -> eda_out/reports/P7g_more_aug.md

Per DFL test match (degraded as production, seed 0), the team GRU trained on the other six with: swap
(P7e, 3 seeds; the reference); + yflip (every training match also mirrored across the long axis, swapped
too); + deg2 (every training match also degraded with seed 1, swapped too); swap with 5 seeds.
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
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P7g_more_aug.md")


def yflip(m):
    s = dict(m)
    s["py"], s["by"] = 1 - np.asarray(m["py"]), 1 - np.asarray(m["by"])
    s["name"] = m["name"] + "+yflip"
    return s


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    clean = [from_dfl(i) for i in ids]
    ms = [degrade(c, dict(rates_for(i), smooth_frames=25), seed=0) for c, i in zip(clean, ids)]
    m2 = [degrade(c, dict(rates_for(i), smooth_frames=25), seed=1) for c, i in zip(clean, ids)]
    sets = {"base": ms, "swap": [swap_teams(m) for m in ms], "yflip": [yflip(m) for m in ms],
            "yflip_swap": [swap_teams(yflip(m)) for m in ms], "deg2": m2, "deg2_swap": [swap_teams(m) for m in m2]}
    T = {k: [features(m) for m in v] for k, v in sets.items()}
    print("features done", flush=True)
    it = lambda m, F: (F, (m["labels"] == pm.H).astype(np.float32), np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32), m["half"] > 0)
    variants = {"swap": ("base", "swap"), "swap_yflip": ("base", "swap", "yflip", "yflip_swap"), "swap_deg2": ("base", "swap", "deg2", "deg2_swap")}
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        m = ms[t]
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        sc = lambda p: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"]}
        for name, keys in variants.items():
            r[name] = sc(fit_predict([it(sets[k][j], T[k][j]) for k in keys for j in tr], (T["base"][t], g)))
        r["swap_5seeds"] = sc(fit_predict([it(sets[k][j], T[k][j]) for k in ("base", "swap") for j in tr], (T["base"][t], g), seeds=(0, 1, 2, 3, 4)))
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["swap", "swap_yflip", "swap_deg2", "swap_5seeds"]
    lines = ["# P7g: takım değişiminin üstüne başka çoğaltmalar (DFL, maç dışarıda bırakarak, sadece takım, sabit ceza)", "", "`analysis/p7g_more_aug.py`.", "",
             "| maç | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", ""]
    lines += ["{}: swap'a fark {:+.3f}, {} / 7 maçta önde.".format(c, (df[c] - df.swap).mean(), int((df[c] > df.swap).sum())) for c in cols[1:]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
