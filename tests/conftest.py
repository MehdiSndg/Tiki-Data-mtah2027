"""Shared test fixtures: synthetic tracking for the preprocessing modules, and a skip marker for
tests that need the gated match bundle in data/."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402

META = asset_paths()["atd_meta"]
needs_data = pytest.mark.skipif(not os.path.isfile(asset_paths()["atd_raw"]),
                                reason="match data not in data/ (gated)")

L, W = 105.3, 68.0          # games.py pitch, metres


def synthetic_kickoff(n, kick_at, second_kick_at=None, shape_from=0):
    """22 slots (11 home, 11 away) holding a kickoff shape, home attacking +x.

    The ball rests on the centre spot with one home player beside it, and from `kick_at` moves
    along +x at 10 m/s. With kick_at=None the teams are mixed across the halfway line, so no
    kickoff shape ever exists. `second_kick_at` re-forms the shape later (an after-goal kickoff).
    """
    rng = np.random.RandomState(0)
    px = np.empty((n, 22), np.float32)
    py = np.empty((n, 22), np.float32)
    home_x = rng.uniform(0.10, 0.45, 11)
    away_x = rng.uniform(0.55, 0.90, 11)
    if kick_at is None:
        home_x, away_x = away_x.copy(), home_x.copy()        # everyone in the wrong half
    home_x[0] = 0.5 - 0.5 / L                                  # kicker 0.5 m behind the spot
    px[:] = np.r_[home_x, away_x]
    py[:] = rng.uniform(0.1, 0.9, 22)
    py[:, 0] = 0.5
    px[:shape_from] = np.r_[away_x, home_x]
    team = np.r_[np.zeros(11), np.ones(11)].astype(np.int8)
    bx = np.full(n, 0.5, np.float32)
    by = np.full(n, 0.5, np.float32)
    for start in [s for s in (kick_at, second_kick_at) if s is not None]:
        k = np.arange(n - start)
        bx[start:] = 0.5 + (k + 1) * (10.0 / 25.0) / L        # 0.4 m per frame
    if second_kick_at is not None:
        bx[second_kick_at - 30:second_kick_at] = 0.5          # ball back on the spot before it
    return px, py, team, bx, by


def straight_line(speed, n, gaps=(), excursion=None):
    """A ball moving along +x at `speed` m/s from x=0.2, y=0.5, normalised 0-1, float32.

    `gaps` are (start, stop) frame ranges set to NaN. `excursion=(start, stop)` throws the ball
    sideways along y at 60 m/s on the frames start..stop-1 and back onto the line at `stop`.
    """
    k = np.arange(n)
    bx = (0.2 + k * (speed / 25.0) / L).astype(np.float32)
    by = np.full(n, 0.5, np.float32)
    if excursion is not None:
        a, b = excursion
        rise = np.minimum(np.arange(1, b - a + 1), np.arange(b - a, 0, -1))  # 1, 2, 1 for 3 frames
        by[a:b] += (rise * 60.0 / 25.0 / W).astype(np.float32)
    for a, b in gaps:
        bx[a:b] = np.nan
        by[a:b] = np.nan
    return bx, by


def first_code(n):
    from mtah.periods import FIRST
    return np.full(n, FIRST, np.int8)
