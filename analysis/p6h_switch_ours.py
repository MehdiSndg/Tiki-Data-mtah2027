"""How the learned team call on our match moves with the Viterbi switch cost. GT-free.

    uv run python analysis/p6h_switch_ours.py      # -> eda_out/reports/P6h_switch_ours.md

The team model trained on all degraded external data (mtah.teammodel.external_model), the v1 chain's
dead ball and periods. Per switch cost: possessions per 90 graded minutes, the ratio to the control
rule's (on degraded DFL the model / control ratio is ~0.86, P6b), Smart Tagging agreement outside the
public window, and the kit-colour judge on frames where the call differs from the control rule's and
the ATD sees the ball. The public window is not read.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402
from mtah.teammodel import decode, external_model, features  # noqa: E402
from mtah.video_match import _hom  # noqa: E402
from p4_common import ours  # noqa: E402
from p5m_kit_team import kit_team_at  # noqa: E402

SWITCHES = (4.0, 8.0, 12.0, 16.0, 24.0)
OUT = os.path.join(ROOT, "eda_out", "reports", "P6h_switch_ours.md")


def main():
    m = from_tactical()
    dead, v1 = ours()
    g = m["half"] > 0
    alive = g & ~dead
    p = external_model().predict_proba(features(m))[:, 1]
    fr = load_frames()
    played = g & ~np.isin(fr["period"].values, NOT_PLAYED)
    minutes = played.sum() / 25 / 60
    runs = pm.valid_runs(np.where(g, pm.D, pm.X).astype(np.int8))
    tag = np.where(fr["tag_H"].values > fr["tag_A"].values, pm.H, np.where(fr["tag_A"].values > fr["tag_H"].values, pm.A, -1))
    outside = np.ones(len(g), bool)
    outside[11185:18685] = False
    judged = alive & outside & (tag >= 0)
    has = ~np.isnan(m["bx"])
    want = alive & has & outside
    kit = kit_team_at(want, np.c_[m["bx"], m["by"]], m, _hom())
    kcode = np.where(kit == 0, pm.H, np.where(kit == 1, pm.A, -1))
    n1 = len(pm.possession_spells(v1, runs))
    lines = ["# P6h: değişim cezası bizim maçta (GT'siz)", "", "`analysis/p6h_switch_ours.py`.", "",
             "| ceza | 90 dk'da possession | model / kontrol | Smart Tagging uyumu | forma hakemiyle uyum (ATD topu olan oyun içi) |",
             "|---|---|---|---|---|"]
    lines.append("| kontrol kuralı | {:.0f} | 1.00 | {:.1%} | {:.1%} |".format(
        n1 / minutes * 90, np.mean(v1[judged] == tag[judged]), np.mean(v1[want & (kcode >= 0)] == kcode[want & (kcode >= 0)])))
    for s in SWITCHES:
        c = np.where(g, decode(p, alive, s), pm.X).astype(np.int8)
        n = len(pm.possession_spells(c, runs))
        k = want & (kcode >= 0)
        lines.append("| {:g} | {:.0f} | {:.2f} | {:.1%} | {:.1%} |".format(
            s, n / minutes * 90, n / n1, np.mean(c[judged] == tag[judged]), np.mean(c[k] == kcode[k])))
        print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
