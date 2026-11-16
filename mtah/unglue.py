"""Metrica's sample ball sits on a player (P9 scan, 13 Nov: within 0.3 m of the nearest player on 53 / 51 / 32% of live
frames raw, 27 / 27 / 14% after our degradation, against 4% for degraded DFL and 5% for our ATD's video-live frames;
the friend's v17 fixed the same thing with noise). unglue(m, seed) moves the ball off the player on the frames where it
is glued (within GLUE_M of the nearest player): a smooth offset (Ornstein-Uhlenbeck in metres, stationary sd SD_M,
correlation time TAU_S) so the ball drifts around the player instead of jumping. Applied to the clean Metrica match
before the degradation; labels untouched.
"""

import numpy as np

FPS = 25
GLUE_M = 0.5
SD_M = 0.8
TAU_S = 1.0


def unglue(m, seed=0, sd=SD_M):
    rng = np.random.default_rng(seed)
    L, W = m["pitch"]
    bx, by = m["bx"].astype(np.float64) * L, m["by"].astype(np.float64) * W
    d = np.hypot(m["px"] * L - bx[:, None], m["py"] * W - by[:, None])
    dmin = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    glued = np.isfinite(dmin) & (dmin < GLUE_M)
    n = len(bx)
    a = np.exp(-1.0 / (TAU_S * FPS))
    e = rng.normal(0, sd * np.sqrt(1 - a * a), (n, 2))
    off = np.zeros((n, 2))
    o = rng.normal(0, sd, 2)
    for t in range(n):
        o = a * o + e[t]
        off[t] = o
    out = dict(m)
    out["bx"] = np.where(glued, (bx + off[:, 0]) / L, m["bx"]).astype(np.float32)
    out["by"] = np.where(glued, (by + off[:, 1]) / W, m["by"]).astype(np.float32)
    return out
