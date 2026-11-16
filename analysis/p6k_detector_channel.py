"""Does a video-detector ball channel help the learned team call? LOMO on degraded DFL.

    python analysis/p6k_detector_channel.py      # -> eda_out/reports/P6k_detector_channel.md

Every match degraded as in P6b (ball smoothed over 25 frames) and given a simulated detector track at
the rates measured for ball_v1 + online tracking on our match (mtah.degrade.add_detector). Two team
models, both with the touch features, trained on the other 6 DFL matches + Metrica 1-2: without and
with the detector features (mtah.teammodel.detector_features). Team call only (dead ball from the
labels), switch cost mtah.teammodel.SWITCH.
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
from mtah.degrade import add_detector, degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.teammodel import SWITCH, decode, fit, team_features  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P6k_detector_channel.md")


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    src = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2)]
    ms, A, B = [], [], []
    for raw, key in src:
        d = add_detector(raw, degrade(raw, dict(rates_for(key), smooth_frames=25), seed=0), seed=0)
        alive = np.isin(d["labels"], (pm.H, pm.A))
        y = (d["labels"] == pm.H).astype(int)
        nodet = {k: v for k, v in d.items() if k not in ("dx", "dy")}
        A.append((team_features(nodet, True)[0], alive, y))
        B.append((team_features(d, True)[0], alive, y))
        ms.append(d)
        print(d["name"], flush=True)
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ms)) if k != t]
        m = ms[t]
        r = {"match": m["name"]}
        for tag, D in (("without", A), ("with", B)):
            p = fit([D[k][0][D[k][1]] for k in tr], [D[k][2][D[k][1]] for k in tr]).predict_proba(D[t][0])[:, 1]
            c = decode(p, D[t][1], SWITCH)
            r[tag] = pm.spell_scores(m["labels"], c, pm.valid_runs(m["labels"]))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    d = df["with"] - df["without"]
    lines = ["# P6k: dedektör kanalı (DFL, maç dışarıda bırakarak)", "", "`analysis/p6k_detector_channel.py`. Sadece takım kararı.", "",
             "| maç | dedektörsüz | dedektörlü |", "|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} |".format(r["match"], r["without"], r["with"]) for r in rows]
    lines += ["", "Ortalama {:.3f} → {:.3f}, fark {:+.3f} ({:+.3f} .. {:+.3f}; {} / {} maçta dedektörlü önde).".format(
        df["without"].mean(), df["with"].mean(), d.mean(), d.min(), d.max(), int((d > 0).sum()), len(d))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
