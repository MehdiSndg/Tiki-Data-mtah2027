"""The clean-owner target (P9eo; the Barca-Hackathon fork's training-only head, +0.007 held out for them): for every
frame of a clean (not degraded) external match, whether a home / an away player is within RADIUS_M of the ball. A
degraded copy blurs exactly this (smoothed, noisy, hidden ball; dropped players), so a second output trained on it
teaches the net to read through the degradation. External data only; never computed on our match.

    y, w = owner_target(raw)        # y (n, 2) float32 [home near, away near]; w (n,) 1 where the clean ball is tracked
"""

import numpy as np

RADIUS_M = 1.5


def owner_target(m, radius=RADIUS_M):
    L, W = m["pitch"]
    dx = (np.asarray(m["px"], np.float64) - np.asarray(m["bx"], np.float64)[:, None]) * L
    dy = (np.asarray(m["py"], np.float64) - np.asarray(m["by"], np.float64)[:, None]) * W
    near = np.hypot(dx, dy) <= radius                       # NaN players compare False
    team = np.asarray(m["slot_team"])
    y = np.stack([near[:, team == 0].any(1), near[:, team == 1].any(1)], 1).astype(np.float32)
    w = (np.isfinite(np.asarray(m["bx"], np.float64)) & np.isfinite(np.asarray(m["by"], np.float64)) & (np.asarray(m["half"]) > 0)).astype(np.float32)
    return y * w[:, None], w
