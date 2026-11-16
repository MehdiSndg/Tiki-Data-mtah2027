"""Dead-ball features, v2: analysis/m1_dead_classifier.features plus restart and look-ahead cues. (step 4)

    from mtah.deadball import features_v2
    X = features_v2(m)          # DataFrame, one row per frame

Added to m1's tracking features (all computable on our ATD, nothing from labels):
  next sighting   where the ball is seen next: distance to the nearest restart spot (corner arcs,
                  penalty spots, centre spot), to a touchline and to a goal line -- after a stoppage
                  the ball comes back at a restart place
  look-ahead      median player speed over the next 2 s against the last 2 s (players walk after a
                  whistle; Borghesi et al. 2023 look 1 s ahead for fouls), and the share of players
                  standing (< 1 m/s) now and in the next 2 s
  kickoff shape   share of each team's players in its own half, against the attack direction
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))

FPS = 25
W2 = 2 * FPS
STAND = 1.0                  # m/s


def _spots(L, W):
    return np.array([[0, 0], [0, W], [L, 0], [L, W], [11, W / 2], [L - 11, W / 2], [L / 2, W / 2]])


def features_v2(m):
    from m1_dead_classifier import features
    f = features(m)
    L, W = m["pitch"]
    n = len(m["bx"])
    bX, bY = m["bx"].astype(float) * L, m["by"].astype(float) * W
    nX, nY = pd.Series(bX).bfill().values, pd.Series(bY).bfill().values
    sp = _spots(L, W)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f["next_spot_d"] = np.min(np.hypot(nX[:, None] - sp[:, 0], nY[:, None] - sp[:, 1]), axis=1)
        f["next_touch_d"] = np.minimum(nY, W - nY)
        f["next_goal_d"] = np.minimum(nX, L - nX)
        X, Y = m["px"].astype(float) * L, m["py"].astype(float) * W
        v = np.full_like(X, np.nan)
        v[5:] = np.hypot(X[5:] - X[:-5], Y[5:] - Y[:-5]) * FPS / 5
        med = np.nanmedian(v, axis=1)
        stand = np.nanmean(np.where(np.isnan(v), np.nan, v < STAND), axis=1)
        past = pd.Series(med).rolling(W2, min_periods=5).median().values
        fut = pd.Series(med[::-1]).rolling(W2, min_periods=5).median().values[::-1]
        f["speed_next_2s"] = fut
        f["speed_ratio"] = np.log((fut + 0.3) / (past + 0.3))
        f["stand_now"] = pd.Series(stand).rolling(FPS, center=True, min_periods=5).mean().values
        f["stand_next_2s"] = pd.Series(stand[::-1]).rolling(W2, min_periods=5).mean().values[::-1]
        sign = np.zeros(n)
        for h, s in m["sign"].items():
            sign[m["half"] == h] = s
        along = (X - L / 2) * sign[:, None]          # + towards the goal home attacks
        home = np.asarray(m["slot_team"]) == 0
        f["own_half_H"] = np.nanmean(np.where(np.isnan(X[:, home]), np.nan, along[:, home] < 1), axis=1)
        f["own_half_A"] = np.nanmean(np.where(np.isnan(X[:, ~home]), np.nan, along[:, ~home] > -1), axis=1)
    return f.fillna(-1)


CONTEXT_S = (4, 20, 40, 80)       # Lang et al. 2022 (stoppages from player tracking): context up to +-80 s


def context(sig, scales_s=CONTEXT_S):
    """Each column's mean over the past s seconds (this frame included) and over the next s seconds (from the next
    frame on), for each scale; -1 where a window is empty."""
    out = {}
    for s in scales_s:
        w = int(s * FPS)
        for c in sig.columns:
            x = sig[c]
            out["{}_past{}".format(c, s)] = x.rolling(w, min_periods=1).mean()
            out["{}_next{}".format(c, s)] = x[::-1].rolling(w, min_periods=1).mean()[::-1].shift(-1)
    return pd.DataFrame(out, index=sig.index).fillna(-1)


def features_v3(m):
    """features_v2 plus long-horizon context (P9f): the median player speed, the share of players standing and
    whether the ball is seen, summarised over 4 / 20 / 40 / 80 s before and after each frame."""
    f = features_v2(m)
    L, W = m["pitch"]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        X, Y = m["px"].astype(float) * L, m["py"].astype(float) * W
        v = np.full_like(X, np.nan)
        v[5:] = np.hypot(X[5:] - X[:-5], Y[5:] - Y[:-5]) * FPS / 5
        sig = pd.DataFrame({"pspeed": np.nanmedian(v, axis=1), "pstand": np.nanmean(np.where(np.isnan(v), np.nan, v < STAND), axis=1),
                            "bseen": np.isfinite(m["bx"].astype(float)).astype(float)}, index=f.index)
    return pd.concat([f, context(sig)], axis=1)


# Broadcast + stillness rule (analysis/p6o_dead_rule.py): the classifier misses a dead ball the ATD keeps
# tracking (it leans on the ball going missing). Found on the public window while checking what "D"
# means there; the thresholds were fixed beforehand, not fitted on it.
AWAY, SLOW, RULE_MIN_BLOCK = 0.8, 1.3, 2 * FPS    # SLOW: the DFL-derived stillness threshold of v0_pipeline


def broadcast_still_dead(fr, broadcast_path):
    """(n,) bool: over a centred 3 s window the broadcast shows no pitch on >= AWAY of the frames it has,
    and the 2 s median player speed (frames["med_speed"]) is below SLOW; blocks under 2 s dropped."""
    b = np.load(broadcast_path)
    away = pd.Series(np.where(b["b_present"], b["b_present"] & ~b["b_valid"], np.nan).astype(float))
    share = away.rolling(3 * FPS, center=True, min_periods=FPS).mean().values
    slow = pd.Series(fr["med_speed"].values).rolling(2 * FPS, center=True, min_periods=10).median().values < SLOW
    rule = (np.nan_to_num(share) >= AWAY) & slow & (fr["half"].values > 0)
    e = np.diff(np.r_[0, rule.astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if z - a < RULE_MIN_BLOCK:
            rule[a:z] = False
    return rule

