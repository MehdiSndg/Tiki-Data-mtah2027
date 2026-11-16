"""The broadcast ATD's ball on tactical frames (the sync is analysis/h_broadcast_sync.py's cache).

    from mtah.broadcast import broadcast_ball_on_tactical
    bx, by = broadcast_ball_on_tactical()        # (n_tactical,) normalised, NaN where none

The broadcast is not aligned to the tactical video; analysis/h_broadcast_sync.py found a constant
offset per half and wrote the tactical -> broadcast frame map to cache/broadcast_on_tactical.npz.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths, source_frames  # noqa: E402

SYNC = os.path.join(ROOT, "cache", "broadcast_on_tactical.npz")


def broadcast_ball():
    """(bx, by) per broadcast frame, normalised pitch coordinates, NaN when absent."""
    n = source_frames("broadcast")
    bx = np.full(n, np.nan, np.float32)
    by = bx.copy()
    with open(asset_paths(source="broadcast")["atd_raw"]) as f:
        for line in f:
            a, _, b = line.rstrip("\n").split(":")
            if b[0] != "N":
                x, y, _ = b.split(",")
                bx[int(a) - 1], by[int(a) - 1] = float(x), float(y)
    return bx, by


def broadcast_ball_on_tactical():
    s = np.load(SYNC)
    bf, present = s["b_frame"], s["b_present"]
    bbx, bby = broadcast_ball()
    ok = present & (bf >= 0) & (bf < len(bbx))
    bx = np.full(len(bf), np.nan, np.float32)
    by = bx.copy()
    bx[ok], by[ok] = bbx[bf[ok]], bby[bf[ok]]
    return bx, by
