"""kin (Group 1): ball kinematics, kick/touch moments, aerial passes, tracking teleports.

The visible ball track is split into segments at gaps and at teleports (steps faster than
`jump_speed_ms`), and each segment is smoothed with Savitzky-Golay (window 7, order 2; as in
Vidal-Codina et al.). A kick/touch candidate is a local maximum of |a| with a direction or speed
change. Its threshold is calibrated PER MATCH by event rate: the strongest candidates are kept so
that every source has `kick_rate_per_min` kicks per minute of visible live play. (A fixed 4 m/s^2
fires on 60% of ATD frames; a per-match p90 still gave 6.6 vs 28.6 kicks/min on BAR-ATH vs DFL,
because clean DFL acceleration is mostly zero during dead ball.) |a| is also exposed as a robust
per-match z-score over moving-ball frames. Metres/seconds throughout: BAR-ATH and IDSSE share
the schema and, through the calibration, the meaning of the kick features.
"""

import numpy as np
from scipy.signal import savgol_filter

from ..context import nan_call
from ..registry import register

DEFAULTS = dict(sg_window=7, sg_order=2, jump_speed_ms=40.0, kick_rate_per_min=12.0,
                moving_speed_ms=1.0, min_dtheta_deg=20.0, min_dspeed_ms=3.0,
                aerial_speed_ms=12.0, aerial_dtheta_deg=5.0, aerial_near_m=2.0,
                horizon_s=10.0, lag=3)


def rate_threshold(values, candidates, minutes, rate):
    """Smallest threshold keeping ~rate*minutes of the strongest candidate values."""
    vals = np.sort(values[candidates & np.isfinite(values)])[::-1]
    if len(vals) == 0:
        return np.inf
    k = int(round(rate * minutes))
    return float(vals[min(max(k, 1), len(vals)) - 1])


def _segments(vis, jump):
    """(segment id per frame (0 = invisible), [(start, stop)] of each visible segment)."""
    starts = vis & (np.concatenate(([True], ~vis[:-1])) | jump)
    seg = np.cumsum(starts) * vis
    out = []
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(seg) != 0) + 1, [len(seg)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if seg[a] > 0:
            out.append((int(a), int(b)))
    return seg, out


def _last_next(flag, n):
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(flag, idx, -1))
    nxt = np.minimum.accumulate(np.where(flag, idx, n)[::-1])[::-1]
    return last, nxt


@register("kin", context=["kin_speed", "kin_acc_rel", "kin_kick", "kin_aerial"])
def kinematics(ctx):
    kc = dict(DEFAULTS, **ctx.cfg["features"].get("kinematics", {}))
    n, fps = ctx.n, ctx.fps
    B, vis = ctx.B, ctx.vis
    lag = int(kc["lag"])

    step = np.full(n, np.nan)
    step[1:] = np.hypot(np.diff(B[:, 0]), np.diff(B[:, 1])) * fps
    jump = np.nan_to_num(step, nan=0.0) > kc["jump_speed_ms"]
    seg, segments = _segments(vis, jump)

    pos = np.full((n, 2), np.nan)
    vel = np.full((n, 2), np.nan)
    acc = np.full((n, 2), np.nan)
    w, o = int(kc["sg_window"]), int(kc["sg_order"])
    for a, b in segments:
        xy = B[a:b].astype(np.float64)
        if b - a >= w:
            pos[a:b] = savgol_filter(xy, w, o, axis=0)
            vel[a:b] = savgol_filter(xy, w, o, deriv=1, delta=1 / fps, axis=0)
            acc[a:b] = savgol_filter(xy, w, o, deriv=2, delta=1 / fps, axis=0)
        elif b - a >= 3:
            pos[a:b] = xy
            vel[a:b] = np.gradient(xy, 1 / fps, axis=0)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    amag = np.hypot(acc[:, 0], acc[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        a_tan = (vel[:, 0] * acc[:, 0] + vel[:, 1] * acc[:, 1]) / speed
        a_norm = np.abs(vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0]) / speed
    jerk = np.full(n, np.nan)
    jerk[1:-1] = np.hypot(acc[2:, 0] - acc[:-2, 0], acc[2:, 1] - acc[:-2, 1]) * fps / 2

    same = np.zeros(n, bool)
    same[lag:-lag] = (seg[2 * lag:] == seg[:-2 * lag]) & (seg[lag:-lag] > 0)
    v_next = np.full((n, 2), np.nan)
    v_prev = np.full((n, 2), np.nan)
    v_next[:-lag], v_prev[lag:] = vel[lag:], vel[:-lag]
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.sum(v_next * v_prev, axis=1) / (np.hypot(*v_next.T) * np.hypot(*v_prev.T))
    dtheta = np.where(same, np.degrees(np.arccos(np.clip(cosang, -1, 1))), np.nan)
    dspeed = np.where(same, np.hypot(*v_next.T) - np.hypot(*v_prev.T), np.nan)

    playing = ctx.data.period > 0
    local_max = np.zeros(n, bool)
    local_max[1:-1] = (amag[1:-1] >= amag[:-2]) & (amag[1:-1] > amag[2:])
    turn = (np.nan_to_num(dtheta) >= kc["min_dtheta_deg"]) | (np.abs(np.nan_to_num(dspeed)) >= kc["min_dspeed_ms"])
    candidates = local_max & turn & playing
    live_minutes = np.sum(vis & playing) / fps / 60.0
    a_kick = rate_threshold(amag, candidates, live_minutes, kc["kick_rate_per_min"])
    kick = candidates & (np.nan_to_num(amag) >= a_kick)
    moving = playing & (np.nan_to_num(speed) >= kc["moving_speed_ms"])
    med = nan_call(np.nanmedian, amag[moving])
    iqr = np.subtract(*nan_call(np.nanpercentile, amag[moving], [75, 25]))
    acc_z = (amag - med) / max(float(iqr), 1e-6)

    horizon = kc["horizon_s"]
    last_k, next_k = _last_next(kick, n)
    t_since = np.where(last_k >= 0, (np.arange(n) - last_k) / fps, horizon)
    t_to = np.where(next_k < n, (next_k - np.arange(n)) / fps, horizon)
    team = ctx.team_sign_near
    dmin = np.fmin(ctx.d_home, ctx.d_away)
    kick_team_prev = np.where(last_k >= 0, team[np.clip(last_k, 0, n - 1)], np.nan)
    kick_team_next = np.where(next_k < n, team[np.clip(next_k, 0, n - 1)], np.nan)
    kick_dmin_prev = np.where(last_k >= 0, dmin[np.clip(last_k, 0, n - 1)], np.nan)
    kick_dmin_next = np.where(next_k < n, dmin[np.clip(next_k, 0, n - 1)], np.nan)

    aerial = ((np.nan_to_num(speed) > kc["aerial_speed_ms"]) & (np.nan_to_num(dtheta, nan=99) < kc["aerial_dtheta_deg"])
              & (np.nan_to_num(dmin, nan=99) < kc["aerial_near_m"]))

    last_j, _ = _last_next(jump & vis, n)
    seg_len = np.zeros(n)
    seg_pos = np.full(n, np.nan)
    for a, b in segments:
        seg_len[a:b] = (b - a) / fps
        seg_pos[a:b] = np.arange(b - a) / max(1, b - a - 1)

    invisible = ~vis
    out = {
        "kin_speed": speed, "kin_acc": amag, "kin_acc_rel": amag / a_kick, "kin_acc_z": acc_z,
        "kin_acc_tan": a_tan, "kin_acc_norm": a_norm, "kin_jerk": jerk,
        "kin_dtheta": dtheta, "kin_dspeed": dspeed,
        "kin_kick": kick.astype(np.float32),
        "kin_t_since_kick": np.minimum(t_since, horizon), "kin_t_to_kick": np.minimum(t_to, horizon),
        "kin_kick_team_prev": kick_team_prev, "kin_kick_team_next": kick_team_next,
        "kin_kick_dmin_prev": kick_dmin_prev, "kin_kick_dmin_next": kick_dmin_next,
        "kin_aerial": np.where(invisible, np.nan, aerial.astype(np.float32)),
        "kin_since_jump_s": np.minimum(np.where(last_j >= 0, (np.arange(n) - last_j) / fps, horizon), horizon),
        "kin_seg_len_s": seg_len, "kin_seg_pos": seg_pos,
    }
    return out
