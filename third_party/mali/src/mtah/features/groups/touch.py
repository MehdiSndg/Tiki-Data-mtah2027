"""tc (Gün 4, Ek M.10): touch sequence -- who touched the ball, in what order, how many times in a row.

Why: on the 103-possession blind pool 87% of the team errors happen with the ball VISIBLE; the
missed H<->A changes are mostly 8-50-frame contested exchanges where the booster sees some
evidence (median P(new) 0.58) and the net erases it. The existing features describe one touch at a
time (kin: last / next kick team, proximity); the official definition of a change is about the
SEQUENCE ("controlled" possession, "more than two consecutive passes"). This group exposes it.

Touch events (25 fps, visible ball, per-segment Savitzky-Golay like kin):
  kick       local max of |a| with a direction / speed change (as kin), rate-calibrated per match
  reception  local min of the tangential acceleration with a speed drop (the ball being stopped),
             rate-calibrated per match the same way
  an event counts only if a player is within `touch_m` of the ball; its team = nearer team at
  that frame (+1 home, -1 away), its margin = |d_away - d_home|.
Columns (home perspective, horizon-capped, NaN-free except where stated)
  tc_last_team / tc_next_team      team of the last event <= t / next event > t (0 if none in horizon)
  tc_last2_team                    team of the event before the last one
  tc_run_prev                      consecutive events by tc_last_team up to t (capped 6)
  tc_run_next                      consecutive events by tc_next_team from the next one (capped 6)
  tc_run_start_since               s since the first event of the current same-team run
  tc_next_run_to                   s until the first event of the next run of the OTHER team
  tc_switch_pending                1 if last and next event teams differ
  tc_switches_3s / tc_events_3s    team switches / events among events within +-1.5 s
  tc_margin_last / tc_margin_next  distance margin of the last / next event (m)
  tc_close_H / tc_close_A          closing speed of the ball to the nearest player of each team
                                   (-d/dt of the distance, m/s, 5-frame smoothed; NaN when hidden)
  tc_close_diff                    tc_close_H - tc_close_A
  tc_duel                          both teams within duel_m of the ball (NaN when hidden)
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from ..registry import register
from .kinematics import _last_next, _segments, rate_threshold

DEFAULTS = dict(sg_window=7, sg_order=2, jump_speed_ms=40.0, lag=3, min_dtheta_deg=20.0,
                min_dspeed_ms=3.0, kick_rate_per_min=12.0, reception_rate_per_min=8.0,
                touch_m=2.0, duel_m=2.0, horizon_s=10.0, window_s=1.5, run_cap=6)


def _events(ctx, kc):
    """(event bool (T,), kick bool, reception bool) on visible, playing frames."""
    n, fps, B, vis = ctx.n, ctx.fps, ctx.B, ctx.vis
    step = np.full(n, np.nan)
    step[1:] = np.hypot(np.diff(B[:, 0]), np.diff(B[:, 1])) * fps
    jump = np.nan_to_num(step, nan=0.0) > kc["jump_speed_ms"]
    seg, segments = _segments(vis, jump)
    vel = np.full((n, 2), np.nan)
    acc = np.full((n, 2), np.nan)
    w, o = int(kc["sg_window"]), int(kc["sg_order"])
    for a, b in segments:
        if b - a >= w:
            xy = B[a:b].astype(np.float64)
            vel[a:b] = savgol_filter(xy, w, o, deriv=1, delta=1 / fps, axis=0)
            acc[a:b] = savgol_filter(xy, w, o, deriv=2, delta=1 / fps, axis=0)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    amag = np.hypot(acc[:, 0], acc[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        a_tan = (vel[:, 0] * acc[:, 0] + vel[:, 1] * acc[:, 1]) / speed
    lag = int(kc["lag"])
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
    minutes = np.sum(vis & playing) / fps / 60.0
    lmax = np.zeros(n, bool)
    lmax[1:-1] = (amag[1:-1] >= amag[:-2]) & (amag[1:-1] > amag[2:])
    turn = (np.nan_to_num(dtheta) >= kc["min_dtheta_deg"]) | (np.abs(np.nan_to_num(dspeed)) >= kc["min_dspeed_ms"])
    cand_k = lmax & turn & playing
    kick = cand_k & (np.nan_to_num(amag) >= rate_threshold(amag, cand_k, minutes, kc["kick_rate_per_min"]))

    dec = -np.nan_to_num(a_tan, nan=0.0)
    lmin = np.zeros(n, bool)
    lmin[1:-1] = (dec[1:-1] >= dec[:-2]) & (dec[1:-1] > dec[2:])
    cand_r = lmin & (np.nan_to_num(dspeed) <= -kc["min_dspeed_ms"]) & playing & ~kick
    recv = cand_r & (dec >= rate_threshold(dec, cand_r, minutes, kc["reception_rate_per_min"]))

    near = np.nan_to_num(np.fmin(ctx.d_home, ctx.d_away), nan=99.0) < kc["touch_m"]
    return (kick | recv) & near, kick & near, recv & near


@register("tc", context=["tc_switches_3s", "tc_close_diff"])
def touch(ctx):
    kc = dict(DEFAULTS, **ctx.cfg["features"].get("touch", {}))
    n, fps = ctx.n, ctx.fps
    ev, _, _ = _events(ctx, kc)
    dh, da = ctx.d_home, ctx.d_away
    team_f = np.where(np.nan_to_num(dh, nan=99) <= np.nan_to_num(da, nan=99), 1.0, -1.0)
    margin_f = np.abs(np.nan_to_num(da, nan=99) - np.nan_to_num(dh, nan=99))
    idx = np.flatnonzero(ev)
    tm = team_f[idx]
    horizon = float(kc["horizon_s"])
    cap = int(kc["run_cap"])
    m = len(idx)

    # run statistics per event: length of the same-team run ending at / starting at each event,
    # and the index of the run's first event
    run_end_len = np.ones(m, np.int64)
    run_first = np.arange(m)
    for i in range(1, m):
        if tm[i] == tm[i - 1]:
            run_end_len[i] = run_end_len[i - 1] + 1
            run_first[i] = run_first[i - 1]
    run_start_len = np.ones(m, np.int64)
    next_other = np.full(m, -1)                    # first later event of the other team
    for i in range(m - 2, -1, -1):
        if tm[i] == tm[i + 1]:
            run_start_len[i] = run_start_len[i + 1] + 1
            next_other[i] = next_other[i + 1]
        else:
            next_other[i] = i + 1

    frames = np.arange(n)
    last_f, next_f = _last_next(ev, n)              # frame of last event <= t, next event >= t
    # next strictly after t
    next_f = np.where(ev, np.r_[next_f[1:], n], next_f)
    pos_last = np.searchsorted(idx, last_f)         # event ordinal (valid where last_f >= 0)
    pos_next = np.searchsorted(idx, next_f)
    has_last = (last_f >= 0) & ((frames - last_f) / fps <= horizon)
    has_next = (next_f < n) & ((next_f - frames) / fps <= horizon)
    pl = np.clip(pos_last, 0, max(m - 1, 0))
    pn = np.clip(pos_next, 0, max(m - 1, 0))
    z = np.zeros(n)
    if m == 0:
        last_team = next_team = last2 = run_prev = run_next = switch = z.copy()
        run_since = next_run_to = np.full(n, horizon)
        marg_last = marg_next = np.full(n, np.nan)
    else:
        last_team = np.where(has_last, tm[pl], 0.0)
        next_team = np.where(has_next, tm[pn], 0.0)
        prev2 = np.clip(pl - 1, 0, m - 1)
        last2 = np.where(has_last & (pl >= 1), tm[prev2], 0.0)
        run_prev = np.where(has_last, np.minimum(run_end_len[pl], cap), 0).astype(float)
        run_next = np.where(has_next, np.minimum(run_start_len[pn], cap), 0).astype(float)
        run_since = np.where(has_last, np.minimum((frames - idx[run_first[pl]]) / fps, horizon), horizon)
        # next run of the other team (relative to the last event's team)
        no = next_other[pl]
        other_f = np.where(no >= 0, idx[np.clip(no, 0, m - 1)], n)
        other_f = np.where(has_last, other_f, np.where(has_next, next_f, n))
        next_run_to = np.minimum(np.where(other_f < n, (other_f - frames) / fps, horizon), horizon)
        next_run_to = np.maximum(next_run_to, 0.0)
        switch = ((last_team != 0) & (next_team != 0) & (last_team != next_team)).astype(float)
        marg_last = np.where(has_last, margin_f[idx[pl]], np.nan)
        marg_next = np.where(has_next, margin_f[idx[pn]], np.nan)

    # events / switches inside +-window
    w = int(round(kc["window_s"] * fps))
    sw_ev = np.zeros(n)
    if m >= 2:
        chg = np.r_[False, tm[1:] != tm[:-1]]
        sw_ev[idx[chg]] = 1.0
    ev_count = uniform_filter1d(ev.astype(float), 2 * w + 1, mode="constant") * (2 * w + 1)
    sw_count = uniform_filter1d(sw_ev, 2 * w + 1, mode="constant") * (2 * w + 1)

    vis = ctx.vis
    close = {}
    for key, d in (("H", dh), ("A", da)):
        dd = np.full(n, np.nan)
        dd[1:-1] = -(d[2:] - d[:-2]) * fps / 2
        dd = np.where(vis, dd, np.nan)
        sm = uniform_filter1d(np.nan_to_num(dd), 5) / np.maximum(uniform_filter1d(np.isfinite(dd).astype(float), 5), 1e-6)
        close[key] = np.where(vis & np.isfinite(dd), np.clip(sm, -30, 30), np.nan)
    duel = np.where(vis, ((np.nan_to_num(dh, nan=99) < kc["duel_m"]) & (np.nan_to_num(da, nan=99) < kc["duel_m"])).astype(float), np.nan)

    return {
        "tc_last_team": last_team, "tc_next_team": next_team, "tc_last2_team": last2,
        "tc_run_prev": run_prev, "tc_run_next": run_next,
        "tc_run_start_since": run_since, "tc_next_run_to": next_run_to,
        "tc_switch_pending": switch, "tc_switches_3s": np.round(sw_count, 6), "tc_events_3s": np.round(ev_count, 6),
        "tc_margin_last": marg_last, "tc_margin_next": marg_next,
        "tc_close_H": close["H"], "tc_close_A": close["A"], "tc_close_diff": close["H"] - close["A"],
        "tc_duel": duel,
    }
