"""v0: a first end-to-end rule pipeline from what the analyses so far support. NOT a submission.

    uv run python analysis/v0_pipeline.py      # writes evaluation/results/v0_rules.csv (gitignored)

Every parameter was fixed before this was run, from physics or external data; none was chosen on
the public window, whose score is printed only as a correctness check:

  periods    algorithmic kickoff (mtah.periods); graded frames before it are D, ungraded are X
  dead ball  at least 2 of 3 signals -- broadcast cut away (no homography), players slow (2 s median
             < 1.3 m/s: DFL), ATD ball missing -- or the ball off the pitch; gaps < 2 s closed,
             blocks < 2 s dropped (a stoppage lasts seconds)
  team       the control rule, R = 1.5 m, T = 0.4 s (best on both DFL and Metrica samples,
             analysis/g4_dfl_rules.py)
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
from g4_dfl_rules import rule_control  # noqa: E402
from mtah.ball import MISSING  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.periods import FIRST, SECOND  # noqa: E402

# Params (fixed in advance; see the docstring)
SLOW = 1.3
CLOSE_GAP, MIN_BLOCK = 50, 50
R, T = 1.5, 0.4
OUT = os.path.join(ROOT, "evaluation", "results", "v0_rules.csv")


def runs(x):
    e = np.diff(np.r_[0, x.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def main():
    df = load_frames()
    b = np.load(os.path.join(ROOT, "cache", "broadcast_on_tactical.npz"))
    play = df["period"].isin([FIRST, SECOND]).values
    slow = pd.Series(df["med_speed"].values).rolling(50, center=True, min_periods=10).median().values < SLOW
    votes = (b["b_present"] & ~b["b_valid"]).astype(int) + slow.astype(int) + (df["ball_status"].values == MISSING).astype(int)
    dead = ((votes >= 2) | df["ball_off_pitch"].values) & play
    for a, z in runs(~dead):
        if z - a < CLOSE_GAP and a > 0 and z < len(dead) and dead[a - 1] and dead[z]:
            dead[a:z] = True
    for a, z in runs(dead):
        if z - a < MIN_BLOCK:
            dead[a:z] = False

    dH = np.nan_to_num(df["H_near_d"].values.astype(float), nan=np.inf)
    dA = np.nan_to_num(df["A_near_d"].values.astype(float), nan=np.inf)
    alive = play & ~dead
    codes = rule_control(dH, dA, alive, R, T)                 # D where not alive
    graded = df["period"].isin([FIRST, SECOND, 3]).values     # PRE2 is graded: it stays D
    codes = np.where(graded, codes, pm.X).astype(np.int8)
    labels = np.array(pm.LABELS)[codes]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(OUT, index=False)

    runs_ = pm.valid_runs(np.where(graded, codes, pm.X).astype(np.int8))
    sp = pm.possession_spells(codes, runs_)
    L = np.array([z - a for a, z, _ in sp]) / 25
    minutes = graded.sum() / 25 / 60
    print("v0 full match (GT-free): {} possessions ({:.1f} per 5 min), mean {:.1f} s, median {:.1f} s, "
          "<2 s {:.0%}, D share {:.0%}".format(len(L), len(L) / (minutes / 5), L.mean(), np.median(L),
                                              (L < 2).mean(), (codes[graded] == pm.D).mean()))
    print("Metrica sample reference: 17.8 per 5 min, mean 10.5 s, median 5.9 s, <2 s 20%, D 38%")
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
