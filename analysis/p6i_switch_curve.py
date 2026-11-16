"""spellF1 and the model / control possession ratio against the switch cost, LOMO on degraded DFL.

    uv run python analysis/p6i_switch_curve.py      # -> eda_out/reports/P6i_switch_curve.md

Same setup as P6b (base features, ball smoothed over 25 frames, dead ball from the labels): for each
test match the model is trained on the other six and decoded at every switch cost. Says how flat
spellF1 is in the switch cost, and which cost gives which possession ratio to the control rule --
the GT-free quantity we can also measure on our match (P6h).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.teammodel import decode, fit, team_features  # noqa: E402

SWITCHES = (2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
OUT = os.path.join(ROOT, "eda_out", "reports", "P6i_switch_curve.md")


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    data = []
    for m in ms:
        F, _ = team_features(m)
        alive = np.isin(m["labels"], (pm.H, pm.A))
        data.append((F, alive, (m["labels"] == pm.H).astype(int)))
    rows = []
    for t, m in enumerate(ms):
        tr = [k for k in range(len(ms)) if k != t]
        p = fit([data[k][0][data[k][1]] for k in tr], [data[k][2][data[k][1]] for k in tr]).predict_proba(data[t][0])[:, 1]
        runs = pm.valid_runs(m["labels"])
        nc = len(pm.possession_spells(control(m, dead="truth"), runs))
        for s in SWITCHES:
            c = decode(p, data[t][1], s)
            rows.append({"match": m["name"], "switch": s, "spell": pm.spell_scores(m["labels"], c, runs)["mean_f1"],
                         "ratio": len(pm.possession_spells(c, runs)) / nc})
        print(m["name"], flush=True)
    df = pd.DataFrame(rows).groupby("switch").agg(spell=("spell", "mean"), lo=("spell", "min"), ratio=("ratio", "mean"))
    lines = ["# P6i: değişim cezası eğrisi (DFL, maç dışarıda bırakarak)", "", "`analysis/p6i_switch_curve.py`.", "",
             "| ceza | ortalama spellF1 | en kötü maç | model / kontrol possession oranı |", "|---|---|---|---|"]
    lines += ["| {:g} | {:.3f} | {:.3f} | {:.2f} |".format(s, r.spell, r.lo, r.ratio) for s, r in df.iterrows()]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
