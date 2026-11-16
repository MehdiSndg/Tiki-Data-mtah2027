"""Reference labelers for the evaluation harness: Match -> per-frame possession_metrics codes.

    from mtah.labelers import baseline_smooth, control, nearest

  nearest          team of the tracked player closest to the ball; D without a ball
  baseline_smooth  the organisers' baseline (starter/baseline.py): nearest, then a centred majority
                   vote over +/- window frames across the whole video, ties to home
  control          a team takes possession when one of its players stays within R m of the ball
                   for T s (analysis/g4_dfl_rules.py); the last team keeps it otherwise
  own_dead         a tracking-only dead-ball estimate: ball missing while the players are slow
                   (2 s median speed < 1.4 m/s, selected on all DFL matches), or the ball off the
                   pitch; gaps < 2 s closed, blocks < 2 s dropped
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
SLOW = 1.4              # m/s; overnight log: best balanced accuracy on 7 DFL matches
MIN_BLOCK = 2 * FPS
OFF_M = 1.0             # metres beyond a line before the ball counts as off the pitch


def _team_distances(m):
    """Per frame, the distance (m) from the ball to the nearest home and away player; inf if none."""
    L, W = m["pitch"]
    d = np.hypot((m["px"].astype(np.float64) - m["bx"][:, None].astype(np.float64)) * L,
                 (m["py"].astype(np.float64) - m["by"][:, None].astype(np.float64)) * W)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dH = np.nanmin(np.where(m["slot_team"] == 0, d, np.nan), axis=1)
        dA = np.nanmin(np.where(m["slot_team"] == 1, d, np.nan), axis=1)
    return np.nan_to_num(dH, nan=np.inf), np.nan_to_num(dA, nan=np.inf)


def _nearest_or_nobody(m):
    """Team code of the nearest player, -1 where there is no ball or no player (as the baseline)."""
    L, W = m["pitch"]
    dx = (m["px"].astype(np.float64) - m["bx"][:, None].astype(np.float64)) * L
    dy = (m["py"].astype(np.float64) - m["by"][:, None].astype(np.float64)) * W
    d2 = np.where(np.isnan(dx) | np.isnan(dy), np.inf, dx * dx + dy * dy)
    k = np.argmin(d2, axis=1)                                   # first slot on a tie, as the baseline
    ok = np.isfinite(d2[np.arange(len(k)), k])
    return np.where(ok, m["slot_team"][k], -1).astype(np.int8)


def nearest(m):
    c = _nearest_or_nobody(m)
    return np.where(c < 0, pm.D, c).astype(np.int8)


def baseline_smooth(m, window):
    c = _nearest_or_nobody(m)
    if window <= 0:
        return np.where(c < 0, pm.D, c).astype(np.int8)
    home = np.concatenate(([0], np.cumsum(c == 0)))
    away = np.concatenate(([0], np.cumsum(c == 1)))
    i = np.arange(len(c))
    lo, hi = np.maximum(0, i - window), np.minimum(len(c), i + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    out = np.full(len(c), pm.D, np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], pm.H, pm.A)
    return out


def _runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def own_dead(m):
    L, W = m["pitch"]
    step = np.hypot(np.diff(m["px"], axis=0) * L, np.diff(m["py"], axis=0) * W) * FPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.r_[np.nan, np.nanmedian(step, axis=1)]
    slow = pd.Series(speed).rolling(MIN_BLOCK, center=True, min_periods=10).median().values < SLOW
    bx, by = m["bx"], m["by"]
    off = (bx < -OFF_M / L) | (bx > 1 + OFF_M / L) | (by < -OFF_M / W) | (by > 1 + OFF_M / W)
    dead = ((np.isnan(bx) & slow) | off) & (m["half"] > 0)
    for a, z in _runs(~dead):
        if z - a < MIN_BLOCK and a > 0 and z < len(dead) and dead[a - 1] and dead[z]:
            dead[a:z] = True
    for a, z in _runs(dead):
        if z - a < MIN_BLOCK:
            dead[a:z] = False
    return dead


def control(m, R=1.5, T=0.4, dead="own"):
    """dead: "truth" (the Match labels' D, to measure the team call alone), "own" (own_dead), or a mask."""
    if isinstance(dead, str) and dead == "truth" and m.get("public_gt"):
        raise ValueError("dead='truth' would take the dead ball from the public ground truth")
    if isinstance(dead, str):
        alive = np.isin(m["labels"], (pm.H, pm.A)) if dead == "truth" else (m["half"] > 0) & ~own_dead(m)
    else:
        alive = (m["half"] > 0) & ~dead
    dH, dA = _team_distances(m)
    need = max(1, int(round(T * FPS)))
    out = np.full(len(dH), pm.D, np.int8)
    cur, runH, runA = None, 0, 0
    for i in range(len(dH)):
        if not alive[i]:
            cur, runH, runA = None, 0, 0
            continue
        runH = runH + 1 if dH[i] <= R and dH[i] <= dA[i] else 0
        runA = runA + 1 if dA[i] <= R and dA[i] < dH[i] else 0
        if runH >= need:
            cur = pm.H
        elif runA >= need:
            cur = pm.A
        if cur is None:                      # restart: first nearest team
            cur = pm.H if dH[i] <= dA[i] else pm.A
        out[i] = cur
    return out
