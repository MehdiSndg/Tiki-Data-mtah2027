"""Bring DFL's possession labels towards Metrica's definition, judged against Metrica's own sample.

    uv run python analysis/g3_dfl_conversion.py

DFL's BallPossession flips on every touch; Metrica's definition ignores an opponent's touch that
"does not have a significant impact" (literature/README.md). Rule tried here:
inside an alive stretch, a spell of one team shorter than X seconds with the OTHER team on both
sides goes back to that team. X is swept; the target is the statistics of the event-derived
labels of Metrica's sample games (analysis/g2_metrica_events.py) -- external to external, the
public ground truth plays no part.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
SWEEP = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 3.0)


def merge_short(codes, max_s):
    """Return a copy with short sandwiched spells absorbed by the team around them."""
    out = codes.copy()
    if max_s <= 0:
        return out
    segs = pm.segments(out, [(0, len(out))])
    for i in range(1, len(segs) - 1):
        a, b, c = segs[i]
        prev_c, next_c = segs[i - 1][2], segs[i + 1][2]
        if c in (pm.H, pm.A) and prev_c == next_c and prev_c in (pm.H, pm.A) and prev_c != c \
                and (b - a) / FPS < max_s:
            out[a:b] = prev_c
    return out


def stats(arrays):
    lengths, dead, frames, d_frames = [], [], 0, 0
    for c in arrays:
        segs = pm.segments(c, [(0, len(c))])
        lengths += [(b - a) / FPS for a, b, k in pm.possession_spells(c, [(0, len(c))])]
        dead += [(b - a) / FPS for a, b, k in segs if k == pm.D]
        frames += len(c)
        d_frames += int((c == pm.D).sum())
    L = np.array(lengths)
    return len(L) / (frames / FPS / 300), L.mean(), np.median(L), (L < 2).mean(), d_frames / frames, np.median(dead)


def main():
    dfl = np.load(os.path.join(ROOT, "cache", "dfl_labels.npz"))
    met = np.load(os.path.join(ROOT, "cache", "metrica_labels.npz"))
    met_arrays = [met[k][met[k] != pm.X] for k in met.files]
    print("{:<26} {:>9} {:>7} {:>7} {:>6} {:>6} {:>12}".format("labels", "spells/5m", "mean s", "med s", "<2s", "D", "dead med s"))
    print("{:<26} {:>9.1f} {:>7.1f} {:>7.1f} {:>6.1%} {:>6.1%} {:>12.1f}".format("Metrica sample (target)", *stats(met_arrays)))
    for x in SWEEP:
        arrays = [merge_short(dfl[k], x) for k in dfl.files]
        print("{:<26} {:>9.1f} {:>7.1f} {:>7.1f} {:>6.1%} {:>6.1%} {:>12.1f}".format("DFL, merge < {:.1f} s".format(x), *stats(arrays)))


if __name__ == "__main__":
    main()
