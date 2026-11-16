"""Timing helpers for the P4 understanding studies: when does a label change, relative to what?

    from mtah.timing import transitions, contact_lag, still_runs, kinks

  transitions  frames where the label passes directly from one team to the other (no D, no X between)
  contact_lag  seconds from the new team's nearest contact onset to a label change (+ = label later)
  still_runs   stretches where the ball moves slower than v_max (m/s, over a centred baseline)
  kinks        frames where the ball turns or changes speed sharply: candidate touches

Positions are normalised (Match format); every speed and distance is in metres via the pitch.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
LOOKBACK_S = 5.0        # contact_lag looks this far before its window for a contact already running
STILL_BASE_S = 0.5      # still_runs: speed over a centred 0.5 s baseline, so detector jitter averages out
KINK_HALF = 3           # kinks: velocity over 3 frames (0.12 s) before and after the frame


def transitions(codes):
    c = np.asarray(codes)
    i = np.flatnonzero((c[1:] != c[:-1]) & np.isin(c[1:], (pm.H, pm.A)) & np.isin(c[:-1], (pm.H, pm.A))) + 1
    return i, c[i].astype(np.int8)


def _team_distance(m, lo, hi, team):
    L, W = m["pitch"]
    sl = slice(lo, hi)
    d = np.hypot((m["px"][sl].astype(np.float64) - m["bx"][sl, None]) * L,
                 (m["py"][sl].astype(np.float64) - m["by"][sl, None]) * W)
    d = np.where(m["slot_team"] == team, d, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(d, axis=1)                       # NaN where no ball or no player of the team


def contact_lag(m, frame, team, R=1.5, window_s=2.0):
    n = len(m["bx"])
    w, back = int(round(window_s * FPS)), int(round(LOOKBACK_S * FPS))
    lo, hi = max(0, frame - w - back), min(n, frame + w + 1)
    near = np.nan_to_num(_team_distance(m, lo, hi, team), nan=np.inf) <= R
    e = np.diff(np.r_[0, near.astype(np.int8), 0])
    onset, end = np.flatnonzero(e == 1) + lo, np.flatnonzero(e == -1) + lo
    onset = onset[(end > frame - w) & (onset <= frame + w)]   # contacts overlapping the window
    if not len(onset):
        return np.nan
    return (frame - onset[np.argmin(np.abs(onset - frame))]) / FPS


def still_runs(bx, by, pitch, v_max=0.5, min_s=1.0):
    L, W = pitch
    x, y = np.asarray(bx, np.float64) * L, np.asarray(by, np.float64) * W
    n, h = len(x), max(1, int(round(STILL_BASE_S * FPS / 2)))
    i = np.arange(n)
    a, b = np.clip(i - h, 0, n - 1), np.clip(i + h, 0, n - 1)
    bad = np.r_[0, np.cumsum(np.isnan(x) | np.isnan(y))]
    clean = (bad[b + 1] - bad[a]) == 0                    # no missing ball anywhere in the baseline
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.hypot(x[b] - x[a], y[b] - y[a]) / ((b - a) / FPS)
    still = clean & (v < v_max)
    e = np.diff(np.r_[0, still.astype(np.int8), 0])
    return [(s, t) for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)) if t - s >= min_s * FPS]


def kinks(bx, by, pitch, min_turn_deg=30.0, min_speed=2.0):
    L, W = pitch
    p = np.c_[np.asarray(bx, np.float64) * L, np.asarray(by, np.float64) * W]
    n, h = len(p), KINK_HALF
    if n < 2 * h + 1:
        return np.zeros(0, int)
    i = np.arange(h, n - h)
    v0, v1 = (p[i] - p[i - h]) * FPS / h, (p[i + h] - p[i]) * FPS / h
    s0, s1 = np.hypot(*v0.T), np.hypot(*v1.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.einsum("ij,ij->i", v0, v1) / (s0 * s1)
        turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        turn = np.where((s0 >= min_speed) & (s1 >= min_speed), turn, 0.0)
        ratio = np.abs(np.log((s1 + 0.5) / (s0 + 0.5))) / np.log(2)     # 1 = speed doubled or halved
        ratio = np.where(np.maximum(s0, s1) >= min_speed, ratio, 0.0)
    score = np.nan_to_num(np.maximum(turn / min_turn_deg, ratio))
    cand = score >= 1
    e = np.diff(np.r_[0, cand.astype(np.int8), 0])
    return np.array([i[s + np.argmax(score[s:t])] for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))], int)
