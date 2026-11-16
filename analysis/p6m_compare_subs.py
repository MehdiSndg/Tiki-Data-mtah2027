"""Two submission files on our match, GT-free: where they differ, who do the judges side with?

    uv run python analysis/p6m_compare_subs.py A.csv B.csv     # -> stdout

Possessions per 90 graded minutes; on frames where the two differ and the ball is alive by both: the
share, and Smart Tagging's side (outside the public window); the kit-colour judge (the kit team of the
person nearest the ATD ball, analysis/p5m_kit_team.kit_team_at) where the ATD sees the ball; and how
many differing frames have no ATD ball (where only the call itself decides).
"""

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
from mtah.frames import load_frames  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402
from mtah.video_match import _hom  # noqa: E402
from p5m_kit_team import kit_team_at  # noqa: E402


def main(a, b):
    A = pm.label_codes(pm.read_label_csv(a))
    B = pm.label_codes(pm.read_label_csv(b))
    m = from_tactical()
    fr = load_frames()
    g = m["half"] > 0
    played = g & ~np.isin(fr["period"].values, NOT_PLAYED)
    minutes = played.sum() / 25 / 60
    runs = pm.valid_runs(np.where(g, pm.D, pm.X).astype(np.int8))
    outside = np.ones(len(g), bool)
    outside[11185:18685] = False
    both = np.isin(A, (pm.H, pm.A)) & np.isin(B, (pm.H, pm.A))
    diff = both & (A != B) & outside
    tag = np.where(fr["tag_H"].values > fr["tag_A"].values, pm.H, np.where(fr["tag_A"].values > fr["tag_H"].values, pm.A, -1))
    st = diff & (tag >= 0)
    has = ~np.isnan(m["bx"])
    kit = kit_team_at(diff & has, np.c_[m["bx"], m["by"]], m, _hom())
    kc = np.where(kit == 0, pm.H, np.where(kit == 1, pm.A, -1))
    k = diff & has & (kc >= 0)
    for name, c in (("A", A), ("B", B)):
        print("{}: {} possessions / 90".format(name, round(len(pm.possession_spells(c, runs)) / minutes * 90)))
    print("differing alive frames {} ({:.1%} of alive-by-both, {:.0%} of them without an ATD ball)".format(
        diff.sum(), diff.sum() / max(1, (both & outside).sum()), np.mean(~has[diff]) if diff.any() else 0))
    print("Smart Tagging on {} of them: A {:.1%}, B {:.1%}".format(st.sum(), np.mean(A[st] == tag[st]), np.mean(B[st] == tag[st])))
    print("kit judge on {} of them: A {:.1%}, B {:.1%}".format(k.sum(), np.mean(A[k] == kc[k]), np.mean(B[k] == kc[k])))


if __name__ == "__main__":
    main(*sys.argv[1:])
