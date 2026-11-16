"""Clean the ATD ball track without ever hiding what was missing.

    b = clean_ball(m["bx"], m["by"], code, half_of(code), attack_sign(meta))
    b["X"], b["Y"], b["speed"], b["status"] ...

Every frame keeps a status code, so a model can always tell an observation from a repair:

    OBSERVED  the ATD's own position, smoothed
    INTERP    a gap of at most SHORT_GAP frames, filled linearly between two observations
    OUTLIER   inside a run of physically impossible steps (> MAX_SPEED); position dropped
    MISSING   no ball, and not repaired -- itself a signal: the ATD drops a resting ball
    MASKED    outside play (warm-up, halftime, before kickoff): the ATD may track other balls

Positions are metres in the attack-direction frame of mtah.players (home side attacks +x).
Smoothing is Savitzky-Golay, window 7, order 2, as Vidal-Codina et al. (2022) do for noisy ball
data, applied per contiguous run so it never smears across a gap; velocity and acceleration are
the same filter's derivatives.
"""

import os
import sys

import numpy as np
from scipy.signal import savgol_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import FPS, PITCH  # noqa: E402
from mtah.periods import FIRST, SECOND  # noqa: E402
from mtah.players import to_attack_frame  # noqa: E402

# Params
OBSERVED, INTERP, OUTLIER, MISSING, MASKED = 0, 1, 2, 3, 4
MAX_SPEED = 35.0             # m/s: no football is kicked harder than this (EDA C-1)
SHORT_GAP = 5                # frames (0.2 s) filled by linear interpolation
SG_WINDOW, SG_ORDER = 7, 2   # Savitzky-Golay (Vidal-Codina et al. 2022)
OFF_PITCH_MARGIN = 0.5       # m beyond the lines before the ball counts as off the pitch
STILL_SPEED = 0.5            # m/s: a stationary ball (EDA B)
STILL_FRAMES = 10            # ... held for at least this many frames


def _runs(mask):
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def clean_ball(bx, by, code, half, sign):
    n = len(bx)
    X, Y = to_attack_frame(bx, by, half, sign)
    play = np.isin(code, (FIRST, SECOND))
    # masked positions are dropped before anything reads them, so warm-up balls cannot leak into
    # the outlier, interpolation or smoothing steps below
    X[~play] = np.nan
    Y[~play] = np.nan
    seen = ~np.isnan(X)
    status = np.where(play, np.where(seen, OBSERVED, MISSING), MASKED).astype(np.int8)

    # outliers: frames strictly inside a run of consecutive too-fast steps
    step = np.full(n, np.nan)
    step[1:] = np.hypot(np.diff(X), np.diff(Y)) * FPS
    fast = np.nan_to_num(step) > MAX_SPEED
    for a, b in _runs(fast):              # steps a..b-1 arrive at frames a..b-1
        status[a:b - 1] = OUTLIER
    X[status == OUTLIER] = np.nan
    Y[status == OUTLIER] = np.nan

    # short gaps between two observations of the same half: linear fill
    for a, b in _runs(status == MISSING):
        if b - a <= SHORT_GAP and a > 0 and b < n and status[a - 1] == OBSERVED and status[b] == OBSERVED \
                and half[a - 1] == half[b]:
            t = np.arange(1, b - a + 1) / (b - a + 1)
            X[a:b] = X[a - 1] + t * (X[b] - X[a - 1])
            Y[a:b] = Y[a - 1] + t * (Y[b] - Y[a - 1])
            status[a:b] = INTERP

    good = (status == OBSERVED) | (status == INTERP)
    vx, vy, ax, ay = (np.full(n, np.nan) for _ in range(4))
    Xs, Ys = np.full(n, np.nan), np.full(n, np.nan)
    for a, b in _runs(good):
        if b - a >= SG_WINDOW:
            for src, dst, v, acc in ((X, Xs, vx, ax), (Y, Ys, vy, ay)):
                dst[a:b] = savgol_filter(src[a:b], SG_WINDOW, SG_ORDER)
                v[a:b] = savgol_filter(src[a:b], SG_WINDOW, SG_ORDER, deriv=1, delta=1.0 / FPS)
                acc[a:b] = savgol_filter(src[a:b], SG_WINDOW, SG_ORDER, deriv=2, delta=1.0 / FPS)
        else:
            Xs[a:b], Ys[a:b] = X[a:b], Y[a:b]
            if b - a >= 2:
                vx[a:b] = np.gradient(X[a:b], 1.0 / FPS)
                vy[a:b] = np.gradient(Y[a:b], 1.0 / FPS)
    speed = np.hypot(vx, vy)

    still = np.zeros(n, bool)
    for a, b in _runs(np.nan_to_num(speed, nan=np.inf) < STILL_SPEED):
        if b - a >= STILL_FRAMES:
            still[a:b] = True

    gap_age = np.full(n, -1, np.int32)
    gap_len = np.full(n, -1, np.int32)
    gap_age[good] = 0
    gap_len[good] = 0
    for a, b in _runs(play & ~good):
        if a > 0 and good[a - 1] and half[a - 1] == half[a]:
            gap_age[a:b] = np.arange(1, b - a + 1)
        gap_len[a:b] = b - a

    off = good & ((np.abs(Xs) > PITCH[0] / 2 + OFF_PITCH_MARGIN) | (np.abs(Ys) > PITCH[1] / 2 + OFF_PITCH_MARGIN))
    return {"X": Xs.astype(np.float32), "Y": Ys.astype(np.float32),
            "vx": vx.astype(np.float32), "vy": vy.astype(np.float32),
            "speed": speed.astype(np.float32), "acc": np.hypot(ax, ay).astype(np.float32),
            "status": status, "off_pitch": off, "stationary": still,
            "gap_age": gap_age, "gap_len": gap_len}
