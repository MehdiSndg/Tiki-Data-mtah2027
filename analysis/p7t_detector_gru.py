"""The simulated video-detector channel (P6k) again, for the team GRU with the team swap. DFL LOMO, team call
only, flat switch, 3 seeds. GPU.

    python analysis/p7t_detector_gru.py      # -> eda_out/reports/P7t_detector_gru.md

P6k tested it on the booster (0.616 -> 0.614). The detector ball is sharp where the ATD's is smoothed, the
one place touches show; the GRU may read it where the booster could not. Every match degraded as production
and given mtah.degrade.add_detector at ball_v1's measured rates; with or without
mtah.teammodel.detector_features (on the swapped matches too).
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
from mtah.degrade import add_detector, degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P7t_detector_gru.md")


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    clean = [from_dfl(i) for i in ids]
    ms = [add_detector(c, degrade(c, dict(rates_for(i), smooth_frames=25), seed=0)) for c, i in zip(clean, ids)]
    sw = [swap_teams(m) for m in ms]
    nodet = lambda m: {k: v for k, v in m.items() if k not in ("dx", "dy")}
    T, Ts = [features(nodet(m)) for m in ms], [features(nodet(m)) for m in sw]
    Td, Tds = [features(m) for m in ms], [features(m) for m in sw]
    print("features done", T[0].shape[1], Td[0].shape[1], flush=True)
    it = lambda m, F: (F, (m["labels"] == pm.H).astype(np.float32), np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32), m["half"] > 0)
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        m = ms[t]
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        sc = lambda p: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"],
             "base": sc(fit_predict([it(ms[k], T[k]) for k in tr] + [it(sw[k], Ts[k]) for k in tr], (T[t], g))),
             "det": sc(fit_predict([it(ms[k], Td[k]) for k in tr] + [it(sw[k], Tds[k]) for k in tr], (Td[t], g)))}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P7t: simüle dedektör kanalı takım değişimli GRU'da (DFL, maç dışarıda bırakarak, sadece takım, 3 tohum)", "", "`analysis/p7t_detector_gru.py`.", "",
             "| maç | dedektörsüz | dedektörlü |", "|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} |".format(r["match"], r["base"], r["det"]) for r in rows]
    lines += ["| ortalama | {:.3f} | {:.3f} |".format(df.base.mean(), df.det.mean()), "",
              "Fark {:+.3f}, {} / 7 maçta önde.".format((df.det - df.base).mean(), int((df.det > df.base).sum()))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
