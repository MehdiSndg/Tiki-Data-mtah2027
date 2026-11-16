"""P4: how much of the touch gain survives a smoothed ball, like our ATD's (7.6 kinks / in-play min)?

    uv run python analysis/p4g_smoothing.py     # -> eda_out/reports/P4g_smoothing.md

Degraded external matches (as in P2), then the ball is smoothed with a centred moving average of w
frames (inside observed stretches only). The touch labelers of analysis/p4f_touch_labeler.py are
scored, dead ball from the labels. The row whose kink rate is closest to our ATD's tells what the
touch idea is worth on the ATD ball, and how much a sharper ball (the video detector) could add back.
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
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.labelers import control  # noqa: E402
from p4f_touch_labeler import SEED, control_touch, spell_f1, touch_only, touches  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "eda_out", "reports", "P4g_smoothing.md")


def smooth(m, w):
    if w <= 1:
        return m
    out = dict(m)
    for k in ("bx", "by"):
        out[k] = pd.Series(m[k].astype(float)).rolling(w, center=True, min_periods=w).mean().values.astype(np.float32)
        out[k] = np.where(np.isnan(m[k]), np.nan, np.where(np.isnan(out[k]), m[k], out[k])).astype(np.float32)
    return out


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    src = {"Metrica": [degrade(from_metrica_sample(g), rates_for("metrica"), seed=SEED) for g in (1, 2)],
           "DFL": [degrade(from_dfl(i), rates_for(i), seed=SEED) for i in ids]}
    lines = ["# P4g: yumuşatılmış topta dokunuş kazancı", "", "`analysis/p4g_smoothing.py`. Bozulmuş dış veri, top w karelik ortalamayla yumuşatıldı; ölü top etiketten.",
             "Dakika = topun görüldüğü oyun içi dakika. Bizim ATD topu: 7.6 kırılma / dakika (P4d).", "",
             "| kaynak | w (kare) | dakikada kırılma | dakikada dokunuş | control | touch | control+touch |", "|---|---|---|---|---|---|---|"]
    from mtah.timing import kinks
    for name, ms in src.items():
        for w in (1, 5, 9, 15, 25):
            r, kr, tr = [], [], []
            for m0 in ms:
                m = smooth(m0, w)
                alive = np.isin(m["labels"], (pm.H, pm.A))
                k = kinks(m["bx"], m["by"], m["pitch"])
                mins = (alive & ~np.isnan(m["bx"])).sum() / FPS / 60         # minutes with a ball, as P4d counts ours
                kr.append(np.sum(alive[k]) / mins)
                tr.append(len(touches(m, alive)[0]) / mins)
                r.append([spell_f1(m, control(m, dead="truth")), spell_f1(m, touch_only(m)), spell_f1(m, control_touch(m))])
            lines.append("| {} | {} | {:.1f} | {:.1f} | {:.3f} | {:.3f} | {:.3f} |".format(name, w, np.mean(kr), np.mean(tr), *np.mean(r, axis=0)))
            print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
