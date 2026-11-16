"""G2: per-frame H/A/D labels derived from Metrica's own sample events, following Metrica's own
event definitions (literature/README.md), and what the tracking shows inside
the dead and alive periods.

    uv run python analysis/g2_metrica_events.py

These labels are OUR derivation from manually annotated events, not Metrica's corrected-tracking
ownership; they are a proxy for the challenge's definition, not ground truth. Rules:

  possession  team of the last PASS, RECOVERY, SET PIECE or SHOT (a BALL LOST hands possession to
              the opponent only through the opponent's RECOVERY that follows it)
  dead starts BALL OUT (at its end frame: the ball crosses the line); a SHOT whose subtype
              contains OUT or GOAL (end frame); FAULT RECEIVED (start); a BALL LOST or BALL OUT
              with a FORCED / VOLUNTARY intervention (start); CARD (start)
  dead ends   the next SET PIECE, a PASS with subtype GOAL KICK, or a RECOVERY (Metrica: "brings
              the ball back into play via a dead-ball situation that is not a SET PIECE")

Output: cache/metrica_labels.npz ({game}_codes per tracking frame, H=0 A=1 D=2 X=3).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

sys.path.insert(0, ROOT)
from mtah.external import metrica_event_labels as labels_for  # noqa: E402

# Params
DATA = os.path.join(ROOT, "data", "external", "metrica-sample-data", "data")
OUT = os.path.join(ROOT, "cache", "metrica_labels.npz")
FPS = 25


def stats(codes):
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    lengths = np.array([b - a for a, b, _ in spells]) / FPS
    dead = np.array([b - a for a, b, c in pm.segments(codes, runs) if c == pm.D]) / FPS
    graded = (codes != pm.X).sum()
    return {"minutes": graded / FPS / 60, "D share": (codes[codes != pm.X] == pm.D).mean(),
            "spells / 5 min": len(lengths) / (graded / FPS / 300), "mean s": lengths.mean(),
            "median s": np.median(lengths), "<2 s": (lengths < 2).mean(), "dead blocks": len(dead),
            "dead median s": np.median(dead)}


def main():
    out = {}
    for g in (1, 2):
        ev = pd.read_csv(os.path.join(DATA, "Sample_Game_{0}/Sample_Game_{0}_RawEventsData.csv".format(g)))
        tr = pd.read_csv(os.path.join(DATA, "Sample_Game_{0}/Sample_Game_{0}_RawTrackingData_Home_Team.csv".format(g)),
                         skiprows=2)
        n = int(tr["Frame"].max())
        codes = labels_for(ev, n)
        out["game{}".format(g)] = codes
        s = stats(codes)
        print("Sample Game {}: ".format(g) + "  ".join("{} {:.2f}".format(k, v) if isinstance(v, float) else
                                                        "{} {}".format(k, v) for k, v in s.items()))
        # ball visibility in the tracking, by derived label
        ball = np.full(n + 1, np.nan)
        ball[tr["Frame"].values] = tr["Ball"].values
        seen = ~np.isnan(ball)
        for name, c in (("H/A", (codes == pm.H) | (codes == pm.A)), ("D", codes == pm.D)):
            print("   ball missing on {} frames: {:.1%}".format(name, (~seen[c]).mean()))
        # missing-ball blocks: how many sit inside a dead period?
        edges = np.diff(np.r_[0, (~seen).astype(int), 0])
        blocks = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
        blocks = [(a, b) for a, b in blocks if b - a >= 25 and codes[a] != pm.X]
        inside = [np.mean(codes[a:b] == pm.D) for a, b in blocks]
        print("   missing-ball blocks >= 1 s: {}; median share of each block that is D: {:.0%}; blocks >= 90% D: {:.0%}".format(
            len(blocks), np.median(inside), np.mean(np.array(inside) >= 0.9)))
    np.savez_compressed(OUT, **out)


if __name__ == "__main__":
    main()
