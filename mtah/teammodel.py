"""Learned team call: per-frame features -> gradient boosting P(home) -> two-state Viterbi. (step 3)

    from mtah.teammodel import team_features, fit, decode
    F = team_features(m)                      # (n, k) float32, NaN allowed
    clf = fit([F1, F2], [y1, y2])             # y: 1 home, 0 away, on alive frames
    codes = decode(clf.predict_proba(F)[:, 1], alive, switch=8.0)

The features look at both teams around the ball, also while the ball is not seen: the ball's last
and next sighting, who was nearest then, and the shape and movement of both teams (the team with the
ball spreads and moves towards the other goal). Windows are centred: the labels are written offline,
so the future is allowed. decode() is a two-state HMM over each alive run with a fixed cost per
change of team (log-odds), the standard cure for over-segmentation (MS-TCN's smoothing, PathCRF's
CRF; literature/README.md).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

FPS = 25
VEL_CENTERED = False         # P9as (external review): speeds as a centred difference instead of the 0.2 s backward one
CAP = 10 * FPS                    # frames: "long ago" for ages and look-aheads
WINDOWS = (FPS, 3 * FPS)          # centred rolling windows, 1 s and 3 s
SEED = 0



def _vel(A, lag, centered=None, fill=np.nan):
    """Velocity per frame (units per second) over `lag` frames of A ((n,) or (n, k)): backward A[t] - A[t - lag], or
    with centered (default VEL_CENTERED) A[t + h] - A[t - h], h = lag // 2. `fill` where it is undefined."""
    centered = VEL_CENTERED if centered is None else centered
    out = np.full(np.shape(A), fill, np.float64)
    if centered:
        h = max(1, lag // 2)
        out[h:-h] = (A[2 * h:] - A[:-2 * h]) * FPS / (2 * h)
    else:
        out[lag:] = (A[lag:] - A[:-lag]) * FPS / lag
    return out

def _roll(x, w, how="mean"):
    s = pd.Series(x)
    r = s.rolling(w, center=True, min_periods=1)
    return getattr(r, how)().values


TOUCH_M = 3.0                     # m: a ball kink this close to a player is that player's touch


def touch_features(m):
    """Touches from the ball's kinks (mtah.timing.kinks: a turn or a change of speed): the team of the
    nearest player within TOUCH_M at each kink, +1 home / -1 away, carried forward and backward with
    its age. 0 where no touch is known."""
    from mtah.timing import kinks
    L, W = m["pitch"]
    n = len(m["bx"])
    k = kinks(m["bx"], m["by"], m["pitch"])
    t = np.zeros(n)
    if len(k):
        d = np.hypot((np.asarray(m["px"], np.float64)[k] - np.asarray(m["bx"], np.float64)[k, None]) * L,
                     (np.asarray(m["py"], np.float64)[k] - np.asarray(m["by"], np.float64)[k, None]) * W)
        d = np.where(np.isnan(d), np.inf, d)
        j = np.argmin(d, axis=1)
        ok = d[np.arange(len(k)), j] <= TOUCH_M
        t[k[ok]] = np.where(np.asarray(m["slot_team"])[j[ok]] == 0, 1, -1)
    idx = np.arange(n, dtype=float)
    has = t != 0
    last = pd.Series(np.where(has, idx, np.nan)).ffill().values
    nxt = pd.Series(np.where(has, idx, np.nan)).bfill().values
    return {"last_touch_team": np.nan_to_num(pd.Series(np.where(has, t, np.nan)).ffill().values),
            "next_touch_team": np.nan_to_num(pd.Series(np.where(has, t, np.nan)).bfill().values),
            "touch_age": np.minimum(np.nan_to_num(idx - last, nan=CAP), CAP),
            "touch_until": np.minimum(np.nan_to_num(nxt - idx, nan=CAP), CAP),
            "touch_r25": _roll(t, FPS), "touch_r75": _roll(t, 3 * FPS)}


def touch_events(m):
    """(n,) +1 / -1 at each kink touched by a home / away player within TOUCH_M (touch_features' events), 0 elsewhere."""
    from mtah.timing import kinks
    L, W = m["pitch"]
    n = len(m["bx"])
    k = kinks(m["bx"], m["by"], m["pitch"])
    t = np.zeros(n)
    if len(k):
        d = np.hypot((np.asarray(m["px"], np.float64)[k] - np.asarray(m["bx"], np.float64)[k, None]) * L,
                     (np.asarray(m["py"], np.float64)[k] - np.asarray(m["by"], np.float64)[k, None]) * W)
        d = np.where(np.isnan(d), np.inf, d)
        j = np.argmin(d, axis=1)
        ok = d[np.arange(len(k)), j] <= TOUCH_M
        t[k[ok]] = np.where(np.asarray(m["slot_team"])[j[ok]] == 0, 1, -1)
    return t


def episode_features(t, windows=(3 * FPS, 10 * FPS)):
    """Touch episodes (both teammates' +0.005 / +0.012): the signed length of the run of same-team touches ending at
    the last touch, and per trailing window the number of touches and of team changes between consecutive touches."""
    n = len(t)
    idx = np.flatnonzero(t != 0)
    run = np.zeros(n)
    change = np.zeros(n)
    r = 0
    for a, i in enumerate(idx):
        r = r + np.sign(t[i]) if a and np.sign(t[i]) == np.sign(t[idx[a - 1]]) else np.sign(t[i])
        run[i] = r
        change[i] = float(a > 0 and np.sign(t[i]) != np.sign(t[idx[a - 1]]))
    has = t != 0
    f = {"touch_run": np.nan_to_num(pd.Series(np.where(has, run, np.nan)).ffill().values)}
    for w in windows:
        f["touch_n_r{}".format(w)] = pd.Series(has.astype(float)).rolling(w, min_periods=1).sum().values
        f["touch_changes_r{}".format(w)] = pd.Series(change).rolling(w, min_periods=1).sum().values
    return f


GK_M = 3.0


def keeper_features(m):
    """Keepers (never read before; P9 audit): distance from the last-seen ball to each keeper, whether the ball was lost
    next to a keeper (+1 home keeper, -1 away; the degradation drops a held ball the same way), and the last-seen
    ball's distance to the goal each team defends."""
    L, W = m["pitch"]
    n = len(m["bx"])
    bX, bY = np.asarray(m["bx"], np.float64) * L, np.asarray(m["by"], np.float64) * W
    seen = ~np.isnan(bX)
    idx = np.arange(n)
    last = pd.Series(np.where(seen, idx, np.nan)).ffill().values
    lX, lY = pd.Series(bX).ffill().values, pd.Series(bY).ffill().values
    X, Y = np.asarray(m["px"], np.float64) * L, np.asarray(m["py"], np.float64) * W
    f = {}
    lost = np.zeros(n)
    for tag, t, val in (("H", 0, 1.0), ("A", 1, -1.0)):
        k = m["gk"].get(t, -1)
        if k is None or k < 0:
            f["last_d_gk" + tag] = np.full(n, CAP)
            continue
        d = np.hypot(X[:, k] - lX, Y[:, k] - lY)
        f["last_d_gk" + tag] = np.nan_to_num(np.minimum(d, CAP), nan=CAP)
        ok = ~seen & ~np.isnan(last)
        at = last[ok].astype(int)
        d_then = np.hypot(X[at, k] - bX[at], Y[at, k] - bY[at])
        lost[np.flatnonzero(ok)[d_then <= GK_M]] = val
    f["lost_at_gk"] = lost
    sign = np.zeros(n)
    for h, s_ in m["sign"].items():
        sign[np.asarray(m["half"]) == h] = s_
    gx_home_defends = np.where(sign >= 0, 0.0, L)            # home attacks +x when sign is +1
    f["goal_d_H"] = np.nan_to_num(np.hypot(lX - gx_home_defends, lY - W / 2), nan=CAP)
    f["goal_d_A"] = np.nan_to_num(np.hypot(lX - (L - gx_home_defends), lY - W / 2), nan=CAP)
    return f


def extra_features(m):
    """keeper_features and episode_features as one (n, k) block, sorted by name (P9n)."""
    f = keeper_features(m)
    f.update(episode_features(touch_events(m)))
    names = sorted(f)
    return np.column_stack([np.asarray(f[k], np.float32) for k in names]), names


def motion_features(m):
    """Where the ball is heading and who is closing in: the team of the nearest player within 25 m
    of the ball's path over the next second (+1 home, -1 away, 0 none; a pass in flight belongs to the
    passer's team, the receiver tells which way), weighted by how directly the ball travels at him; and
    per team, the players within 15 m running at the ball faster than 2 m/s."""
    L, W = m["pitch"]
    X, Y = np.asarray(m["px"], np.float64) * L, np.asarray(m["py"], np.float64) * W
    bX, bY = np.asarray(m["bx"], np.float64) * L, np.asarray(m["by"], np.float64) * W
    team = np.asarray(m["slot_team"])
    n = len(bX)
    lag = 5
    bvx, bvy = _vel(bX, lag), _vel(bY, lag)
    vx, vy = _vel(X, lag), _vel(Y, lag)
    f = {}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dx, dy = X - bX[:, None], Y - bY[:, None]
        d = np.hypot(dx, dy)
        bs = np.hypot(bvx, bvy)
        cos = (dx * bvx[:, None] + dy * bvy[:, None]) / (d * bs[:, None])          # ball heading at the player
        ahead = (cos > 0.8) & (d < 25) & (bs[:, None] > 3)
        score = np.where(ahead, cos / np.maximum(d, 1.0), -np.inf)
        k = np.argmax(score, axis=1)
        ok = np.isfinite(score[np.arange(n), k])
        f["ball_to_team"] = np.where(ok, np.where(team[k] == 0, 1.0, -1.0) * score[np.arange(n), k] * 10, 0.0)
        app = -(dx * vx + dy * vy) / np.maximum(d, 0.5)                             # closing speed towards the ball
        run = (app > 2) & (d < 15)
        f["chase_H"] = np.sum(run & (team == 0), axis=1).astype(float)
        f["chase_A"] = np.sum(run & (team == 1), axis=1).astype(float)
        f["chase_diff"] = f["chase_H"] - f["chase_A"]
    for w in WINDOWS:
        f["ball_to_team_r{}".format(w)] = _roll(f["ball_to_team"], w)
        f["chase_diff_r{}".format(w)] = _roll(f["chase_diff"], w)
    return f


def carry_features(m):
    """Dribbling: the nearest player within 3 m moving with the ball (relative speed < 2 m/s while the
    ball moves > 1.5 m/s): carry_team +1 home / -1 away / 0; and how long (s, capped at 10) the same
    player has been the one nearest within 3 m, signed by his team."""
    L, W = m["pitch"]
    X, Y = np.asarray(m["px"], np.float64) * L, np.asarray(m["py"], np.float64) * W
    bX, bY = np.asarray(m["bx"], np.float64) * L, np.asarray(m["by"], np.float64) * W
    team = np.asarray(m["slot_team"])
    n = len(bX)
    lag = 5
    bvx, bvy = _vel(bX, lag), _vel(bY, lag)
    vx, vy = _vel(X, lag), _vel(Y, lag)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.hypot(X - bX[:, None], Y - bY[:, None])
        d = np.where(np.isnan(d), np.inf, d)
        k = np.argmin(d, axis=1)
        r = np.arange(n)
        near = d[r, k] < 3
        rel = np.hypot(bvx - vx[r, k], bvy - vy[r, k])
        sign = np.where(team[k] == 0, 1.0, -1.0)
        carry = near & (rel < 2) & (np.hypot(bvx, bvy) > 1.5)
    run = np.zeros(n)
    for t in range(1, n):
        if near[t]:
            run[t] = run[t - 1] + 1 if (near[t - 1] and k[t] == k[t - 1]) else 1
    f = {"carry_team": np.where(carry, sign, 0.0), "same_holder_s": np.where(near, sign * np.minimum(run / FPS, 10), 0.0)}
    for w in WINDOWS:
        f["carry_team_r{}".format(w)] = _roll(f["carry_team"], w)
    return f


RUN_SPEED = 7.0                   # m/s: a player sprinting to a loose ball
REACH_T = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def reach_features(m):
    """Who gets to the ball first: the ball carried on at its current velocity for REACH_T seconds; at
    each point, each team's best arrival slack (ball time minus the player's run time at RUN_SPEED).
    reach_team is the team (+1 home / -1 away) that can be there first at the earliest point anyone can,
    reach_margin how much sooner (s), 0 when no one can."""
    L, W = m["pitch"]
    X, Y = np.asarray(m["px"], np.float64) * L, np.asarray(m["py"], np.float64) * W
    bX, bY = np.asarray(m["bx"], np.float64) * L, np.asarray(m["by"], np.float64) * W
    team = np.asarray(m["slot_team"])
    n = len(bX)
    lag = 5
    bvx, bvy = _vel(bX, lag, fill=0.0), _vel(bY, lag, fill=0.0)
    best = {0: np.full((n, len(REACH_T)), -np.inf), 1: np.full((n, len(REACH_T)), -np.inf)}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j, t in enumerate(REACH_T):
            px, py = bX + bvx * t, bY + bvy * t
            slack = t - np.hypot(X - px[:, None], Y - py[:, None]) / RUN_SPEED
            for k in (0, 1):
                best[k][:, j] = np.nanmax(np.where(team == k, slack, -np.inf), axis=1)
        can = np.maximum(best[0], best[1]) >= 0
        first = np.argmax(can, axis=1)
        anyone = can.any(axis=1) & ~np.isnan(bX)
        r = np.arange(n)
        h, a = best[0][r, first], best[1][r, first]
        reach = np.where(anyone, np.where(h >= a, 1.0, -1.0), 0.0)
        margin = np.where(anyone, np.abs(np.nan_to_num(h, neginf=-5) - np.nan_to_num(a, neginf=-5)), 0.0)
    f = {"reach_team": reach, "reach_margin": np.minimum(margin, 5.0), "reach_signed": reach * np.minimum(margin, 5.0)}
    for w in WINDOWS:
        f["reach_signed_r{}".format(w)] = _roll(f["reach_signed"], w)
    return f


def detector_features(m):
    """The same nearest-team view as the ATD ball's, from a video detector's track m["dx"], m["dy"]."""
    L, W = m["pitch"]
    X, Y = m["px"].astype(np.float64) * L, m["py"].astype(np.float64) * W
    dX, dY = np.asarray(m["dx"], np.float64) * L, np.asarray(m["dy"], np.float64) * W
    team = np.asarray(m["slot_team"])
    f = {}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for tag, x, y in (("det", dX, dY), ("det_last", pd.Series(dX).ffill().values, pd.Series(dY).ffill().values),
                          ("det_next", pd.Series(dX).bfill().values, pd.Series(dY).bfill().values)):
            d = np.hypot(X - x[:, None], Y - y[:, None])
            dh = np.nanmin(np.where(team == 0, d, np.nan), axis=1)
            da = np.nanmin(np.where(team == 1, d, np.nan), axis=1)
            f[tag + "_dH"], f[tag + "_dA"], f[tag + "_diff"] = dh, da, dh - da
    seen = ~np.isnan(dX)
    f["det_seen"] = seen.astype(float)
    f["det_seen_atd_missing"] = (seen & np.isnan(np.asarray(m["bx"], float))).astype(float)
    for w in WINDOWS:
        f["det_diff_r{}".format(w)] = _roll(f["det_diff"], w)
    return f


def team_features(m, touches=False, motion=False, reach=False, carry=False):
    L, W = m["pitch"]
    n = len(m["bx"])
    X, Y = m["px"].astype(np.float64) * L, m["py"].astype(np.float64) * W
    bX, bY = m["bx"].astype(np.float64) * L, m["by"].astype(np.float64) * W
    seen = ~np.isnan(bX)
    idx = np.arange(n)
    team = np.asarray(m["slot_team"])
    home, away = team == 0, team == 1
    # attack direction of HOME along x, per frame (0 outside the halves)
    sign = np.zeros(n)
    for h, s in m["sign"].items():
        sign[m["half"] == h] = s

    last = pd.Series(np.where(seen, idx, np.nan)).ffill().values
    nxt = pd.Series(np.where(seen, idx, np.nan)).bfill().values
    age = np.minimum(np.nan_to_num(idx - last, nan=CAP), CAP)
    until = np.minimum(np.nan_to_num(nxt - idx, nan=CAP), CAP)
    lX, lY = pd.Series(bX).ffill().values, pd.Series(bY).ffill().values
    nX, nY = pd.Series(bX).bfill().values, pd.Series(bY).bfill().values
    f = {"seen": seen.astype(float), "age": age, "until": until}

    lag = 5
    vx, vy = _vel(X, lag), _vel(Y, lag)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for tag, bx_, by_ in (("now", bX, bY), ("last", lX, lY), ("next", nX, nY)):
            d = np.hypot(X - bx_[:, None], Y - by_[:, None])
            dh = np.nanmin(np.where(home, d, np.nan), axis=1)
            da = np.nanmin(np.where(away, d, np.nan), axis=1)
            f[tag + "_dH"], f[tag + "_dA"] = dh, da
            f[tag + "_diff"] = dh - da
            for r in (3.0, 10.0):
                f["{}_nH{:.0f}".format(tag, r)] = np.sum((d <= r) & home, axis=1)
                f["{}_nA{:.0f}".format(tag, r)] = np.sum((d <= r) & away, axis=1)
            if tag == "now":
                # ball speed and the nearest home/away player's speed towards the ball
                bsp = np.r_[np.nan, np.hypot(np.diff(bX), np.diff(bY)) * FPS]
                f["ball_speed"] = np.minimum(bsp, 40)
        # what the ball did at its last and next sighting, as the nearest team then
        for tag in ("last", "next"):
            src = last if tag == "last" else nxt
            ok = ~np.isnan(src)
            v = np.full(n, np.nan)
            v[ok] = f["now_diff"][src[ok].astype(int)]
            f[tag + "_seen_diff"] = v
        # team shape and movement, home-relative, signed so + is towards the goal HOME attacks
        for tag, mask in (("H", home), ("A", away)):
            cx = np.nanmean(np.where(mask, X, np.nan), axis=1)
            f["cx_" + tag] = (cx - L / 2) * sign
            f["sx_" + tag] = np.nanstd(np.where(mask, X, np.nan), axis=1)
            f["sy_" + tag] = np.nanstd(np.where(mask, Y, np.nan), axis=1)
            f["vx_" + tag] = np.nanmean(np.where(mask, vx, np.nan), axis=1) * sign
            f["n_" + tag] = np.sum(mask & ~np.isnan(X), axis=1)
        f["ball_x"] = (pd.Series(bX).ffill().bfill().values - L / 2) * sign
        f["ball_y"] = np.abs(pd.Series(bY).ffill().bfill().values - W / 2)
    for w in WINDOWS:
        for k in ("now_diff", "vx_H", "vx_A", "sx_H", "sx_A", "cx_H", "cx_A"):
            f["{}_r{}".format(k, w)] = _roll(f[k], w)
        f["seen_r{}".format(w)] = _roll(f["seen"], w)
    if touches:
        f.update(touch_features(m))
    if "dx" in m:
        f.update(detector_features(m))
    if motion:
        f.update(motion_features(m))
    if reach:
        f.update(reach_features(m))
    if carry:
        f.update(carry_features(m))
    names = sorted(f)
    return np.column_stack([np.asarray(f[k], np.float32) for k in names]), names


class _Ensemble:
    """The mean P(home) of several boosters (predict_proba like one)."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def fit(features, targets, big=False, ensemble=0):
    """ensemble > 0: that many boosters, each on 70% of the features (sklearn max_features) with its
    own seed, averaged."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xs = np.concatenate(features)
    ys = np.concatenate(targets)
    if ensemble:
        return _Ensemble([HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=63, max_features=0.7,
                                                         random_state=SEED + i).fit(Xs, ys) for i in range(ensemble)])
    if big:
        return HistGradientBoostingClassifier(max_iter=800, learning_rate=0.04, max_leaf_nodes=127, min_samples_leaf=200,
                                              l2_regularization=1.0, random_state=SEED).fit(Xs, ys)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
                                          random_state=SEED).fit(Xs, ys)


def decode(p_home, alive, switch):
    """(n,) codes: H/A on alive frames from a two-state Viterbi per alive run, D elsewhere.

    switch: the cost of a change of team, a scalar or one value per frame (the cost of changing at it)."""
    eps = 1e-4
    p = np.clip(np.asarray(p_home, np.float64), eps, 1 - eps)
    lh, la = np.log(p), np.log(1 - p)
    sw = np.broadcast_to(np.asarray(switch, np.float64), p.shape)
    out = np.full(len(p), pm.D, np.int8)
    e = np.diff(np.r_[0, alive.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        k = b - a
        back = np.zeros((k, 2), np.int8)
        sh, sa = lh[a], la[a]
        for t in range(1, k):
            c = sw[a + t]
            hh, ah = sh, sa - c                 # into H from H, from A
            aa, ha = sa, sh - c                 # into A from A, from H
            back[t, 0] = 0 if hh >= ah else 1
            back[t, 1] = 1 if aa >= ha else 0
            sh = max(hh, ah) + lh[a + t]
            sa = max(aa, ha) + la[a + t]
        s = 0 if sh >= sa else 1
        path = np.empty(k, np.int8)
        for t in range(k - 1, -1, -1):
            path[t] = s
            s = back[t, s]
        out[a:b] = np.where(path == 0, pm.H, pm.A)
    return out


SWITCH = 12.0                     # P6b nested LOMO, variant touch_aug_seeds: 12 on 6 of 7 test matches (8 on one);
                                  # spellF1 is flat over 8-16 (P6i)
SMOOTH = 25                       # frames: the degraded external ball smoothed like our ATD's (mtah.degrade)
TOUCHES = True                    # touch_features: +0.009 on DFL LOMO (P6b touch vs base)
MOTION = True                     # motion_features: +0.016 (touch -> touch_motion), +0.023 with the copies (0.647)
# extra degraded copies of every training match: (smooth_frames, seed). An unsmoothed ball and two more
# degradation seeds: 0.610 -> 0.624 on DFL LOMO together with the touches (P6b touch_aug_seeds)
COPIES = ((0, 0), (SMOOTH, 1), (SMOOTH, 2))


def features(m):
    """team_features with the production options."""
    return team_features(m, TOUCHES, MOTION)[0]


def external_model(smooth=SMOOTH):
    """The production team model, trained on the 7 DFL matches, each degraded with (smooth, seed 0) and
    every COPIES setting. Cached in cache/ under a
    hash of the code it depends on (this file, mtah/degrade.py, mtah/external.py, mtah/timing.py)."""
    import hashlib
    import pickle
    h = hashlib.sha256(str(smooth).encode())
    for f in ("teammodel.py", "degrade.py", "external.py", "timing.py"):
        h.update(open(os.path.join(ROOT, "mtah", f), "rb").read())
    path = os.path.join(ROOT, "cache", "team_model_{}.pkl".format(h.hexdigest()[:16]))
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    clf = _train_external(smooth)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(clf, fh)
    return clf


def external_training_matches(smooth=SMOOTH, copies=COPIES):
    """The degraded DFL matches the production models train on: each with (smooth, seed 0) and every copy."""
    import glob
    from mtah.degrade import degrade, rates_for
    from mtah.external import DFL, from_dfl
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    out = []
    for i in ids:
        raw = from_dfl(i)
        for sm, sd in ((smooth, 0),) + tuple(copies):
            out.append(degrade(raw, dict(rates_for(i), smooth_frames=sm), seed=sd))
    return out


def _train_external(smooth):
    import glob
    from mtah.degrade import degrade, rates_for
    from mtah.external import DFL, from_dfl
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    # DFL only: the configuration P6b validated (0.647); Metrica 1-3 added to it gave 0.643 (P6b met)
    src = [(from_dfl(i), i) for i in ids]
    Fs, ys = [], []
    for raw, key in src:
        for sm, sd in ((smooth, 0),) + COPIES:
            e = degrade(raw, dict(rates_for(key), smooth_frames=sm), seed=sd)
            alive = np.isin(e["labels"], (pm.H, pm.A))
            Fs.append(features(e)[alive])
            ys.append((e["labels"][alive] == pm.H).astype(int))
    return fit(Fs, ys)


def decode3(p_home, p_dead, graded, switch, dead_cost):
    """(n,) codes H/A/D on graded frames from one three-state Viterbi per graded run: the dead ball and
    the team decided together. Emissions: log P(dead), log(1 - P(dead)) + log P(home | alive) and its
    away twin; a change between H and A costs `switch`, entering or leaving D costs `dead_cost`."""
    eps = 1e-4
    ph = np.clip(np.asarray(p_home, np.float64), eps, 1 - eps)
    pd_ = np.clip(np.asarray(p_dead, np.float64), eps, 1 - eps)
    em = np.stack([np.log1p(-pd_) + np.log(ph), np.log1p(-pd_) + np.log1p(-ph), np.log(pd_)], axis=1)
    T = np.array([[0, -switch, -dead_cost], [-switch, 0, -dead_cost], [-dead_cost, -dead_cost, 0]])
    out = np.full(len(ph), pm.X, np.int8)
    e = np.diff(np.r_[0, np.asarray(graded).astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        k = b - a
        back = np.zeros((k, 3), np.int8)
        score = em[a].copy()
        for t in range(1, k):
            cand = score[:, None] + T                 # from i (rows) to j (cols)
            back[t] = np.argmax(cand, axis=0)
            score = cand[back[t], np.arange(3)] + em[a + t]
        s = int(np.argmax(score))
        path = np.empty(k, np.int8)
        for t in range(k - 1, -1, -1):
            path[t] = s
            s = back[t, s]
        out[a:b] = np.array([pm.H, pm.A, pm.D], np.int8)[path]
    return out
