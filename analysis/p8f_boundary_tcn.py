"""The boundary model as a GRU + TCN mean (v10 uses a GRU alone). DFL LOMO, team call (dead ball from the
labels), v10's team side, fresh seeds (3, 4, 5). GPU.

    python analysis/p8f_boundary_tcn.py      # -> eda_out/reports/P8f_boundary_tcn.md
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
from p7d_player_set import fit_predict  # noqa: E402

SEEDS = (3, 4, 5)
OUT = os.path.join(ROOT, "eda_out", "reports", "P8f_boundary_tcn.md")


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    ms += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    sw = [swap_teams(m) for m in ms]
    T, Ts = [features(m) for m in ms], [features(m) for m in sw]
    S = [G.short_mask(m["labels"]) for m in ms]
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    rows = []
    for t in range(len(ids)):
        ks = [k for k in range(len(ms)) if k != t]
        tt = [(F, (m["labels"] == pm.H).astype(np.float32), aw(m) * np.where(S[k], 2.0, 1.0).astype(np.float32), m["half"] > 0)
              for k in ks for m, F in ((ms[k], T[k]), (sw[k], Ts[k]))]
        tb = [(F, G.boundary_target(m["labels"]), aw(m), m["half"] > 0) for k in ks for m, F in ((ms[k], T[k]), (sw[k], Ts[k]))]
        m = ms[t]
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        p = (fit_predict(tt, (T[t], g), seeds=SEEDS) + fit_predict(tt, (T[t], g), make_net=G._tcn, seeds=SEEDS)) / 2
        bg = fit_predict(tb, (T[t], g), seeds=SEEDS)
        bt = fit_predict(tb, (T[t], g), make_net=G._tcn, seeds=SEEDS)
        sc = lambda b: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, G.SWITCH_HI * (1 - b) + G.SWITCH_LO * b), pm.X).astype(np.int8),
                                       pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"], "flat": pm.spell_scores(m["labels"], np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"],
             "b_gru": sc(bg), "b_tcn": sc(bt), "b_mean": sc((bg + bt) / 2)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["flat", "b_gru", "b_tcn", "b_mean"]
    lines = ["# P8f: sınır modeli GRU + TCN (v10 takım tarafı, ölü top etiketten, DFL, tohum 3–5)", "", "`analysis/p8f_boundary_tcn.py`.", "",
             "| maç | sabit ceza | sınır GRU (v10) | sınır TCN | sınır ort. |", "|---|---|---|---|---|"]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", ""]
    lines += ["{}: sınır GRU'ya fark {:+.3f}, {} / 7 maçta önde.".format(c, (df[c] - df.b_gru).mean(), int((df[c] > df.b_gru).sum())) for c in ("flat", "b_tcn", "b_mean")]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
