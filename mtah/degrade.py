"""Degrade clean external tracking the way our automatic ATD is degraded, at measured rates.

    from mtah.degrade import degrade
    noisy = degrade(m, seed=0)                  # a new Match; m is not modified

Every rate comes from a GT-free measurement on our match (notes/VERI.md,
notes/BULGULAR.md); the labels are never changed. Steps, in order:

  1. view_window          the tactical camera shows a moving window along x, kept on the pitch;
                          outfield players and the ball outside it are gone, a keeper outside it
                          is held where last seen (the ATD does this: 93% present, ~half off-screen)
  2. drop_stationary_ball a still ball is dropped (and the ball of some whole dead-ball stretches)
  2b. lose_ball           the ball is lost for a few seconds inside the view
  3. drop_gk_hold         a ball held by a keeper is dropped
  4. ball_drift           at some gap starts the ball is extrapolated on its last velocity; some
                          stretches carry a 1-5 m offset
  5. fragment_players     identities are gone: tracks are cut into pieces placed in free slots,
                          46 slots for 22 players, and some pieces are lost
  6. double_count         some stretches show a shifted copy of a player
  7. freeze_keepers       a keeper slot freezes while the keeper moves on (the "ghost" keeper)

The view is a window along x only; the trapezoid of the real view is not modelled.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation"))
import possession_metrics as pm  # noqa: E402

FPS = 25
VIEW_MARGIN = 0.02                              # the view may show this much beyond each goal line

DEFAULT_RATES = {
    # 1. camera: window width / pitch length p5, p50, p95; the ball's place in it (fraction of width)
    "view_width": (0.44, 0.72, 0.96),           # degradation_model.md, Kamera
    "view_ball_pos": (0.19, 0.77),              # degradation_model.md, Kamera (p10 .. p90)
    "view_step_s": 10.0,                        # the camera's framing changes over seconds
    # 2. still ball
    "still_speed": 0.5,                         # m/s; EDA B: 88% of YOLO balls in ATD gaps are this slow
    "still_frames": 50,                         # calibration (eda_out/reports/P2_calibration.md): our ATD keeps a ball
                                                # still for > 1 s while seen (p90 1.2 s) and has few gaps under 2 s
    "dead_ball_drop": 0.0,                      # share of dead-ball stretches dropped: 0, the ATD follows a moving
                                                # dead ball (degradation_model.md) and a still one goes by still_frames
    # 2b. ball lost inside the view (EDA C-1: most losses are not at the frame edge)
    "loss_per_min": 1.0,                        # calibration on DFL: in-play missing 12.1% vs our 12.2% (v0 estimate)
    "loss_median_s": 5.0,                       # same: mean ~7.5 s
    "loss_sigma": 0.8,
    # 3. keeper holding the ball
    "gk_hold_m": 1.0,
    "gk_hold_s": 0.5,
    # 4. drift and offset
    "drift_p": 0.025,                           # b_atd_ball_audit.py: ~2-3% of frames drift > 5 m
    "drift_s": (1.0, 2.0),
    "jitter_share": 0.06,                       # b_atd_ball_audit.py: ~6% of frames 1-5 m off
    "jitter_m": (1.0, 5.0),
    # 5. track fragments
    "piece_median_s": 35.0,                     # c_tracking_quality.py: unbroken track median 35 s
    "piece_drop": 0.1,                          # calibration: players-per-team p5 within 0.5 of ours
    # 6. double counting
    "dup_share": 0.5,                           # calibration: gives our 7% of frames with > 10 outfield players
                                                # (p1_report.py's measure; a copy only shows there when the
                                                # whole team is in view, so the copy rate is much higher)
    "dup_m": (0.5, 1.5),
    # 7. ghost keeper
    "smooth_frames": 0,                         # off: P2 numbers stay as they are. On degraded DFL, 25 gives a frame-to-
                                                # frame speed change of 0.18 m/s (our ATD 0.20) but 12.9 kinks / min
                                                # (ATD 7.6): no single width matches both (notes/BULGULAR.md)
    "gk_freeze_share": 0.15,                    # overnight log, morning: 19.9% (H) and 9.6% (A) of frames in view
    "gk_freeze_mean_s": 4.5,                    # same: ~6.3 s (H) and ~2.9 s (A) per stretch
    # 8. a dead ball the ATD keeps tracking: DFL freezes its ball while dead, so after the still-ball step
    # no dead frame keeps a ball, while our ATD sees one on ~13% of dead frames (dead ball v3, 26 Oct),
    # moving (median ~2.7 m/s: carried to the restart). Off by default so earlier numbers stay.
    "dead_carry_p": 0.0,
}


# Per-source overrides (calibration): Metrica's own tracking already loses the ball in play (5.1%
# clean, 10.9% degraded without extra losses, ours 12.2%), so it gets no extra losses.
SOURCE_RATES = {"metrica": {"loss_per_min": 0.0},
                # SkillCorner's broadcast ball is already off camera on ~27% of frames before the view step
                "sc": {"loss_per_min": 0.0}}


def rates_for(name):
    r = dict(DEFAULT_RATES)
    for prefix, over in SOURCE_RATES.items():
        if name.startswith(prefix):
            r.update(over)
    return r


def harsh_rates(name, smooth=50):
    """A harsher degradation than rates_for(name) (P9e, the teammate's mixed-degradation training): ball losses,
    jitter, drift and track pieces x1.5 and the ball smoothed over 50 frames."""
    r = rates_for(name)
    return dict(r, loss_per_min=r["loss_per_min"] * 1.5, jitter_share=r["jitter_share"] * 1.5, drift_p=r["drift_p"] * 1.5,
                piece_drop=min(1.0, r["piece_drop"] * 1.5), smooth_frames=smooth)


def _runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def _copy(m, **arrays):
    out = dict(m)
    out.update(arrays)
    return out


def _knots(n, step, rng, draw):
    """A smooth series: values drawn every `step` frames, linear in between."""
    at = np.arange(0, n + step, step)
    return np.interp(np.arange(n), at, [draw(rng) for _ in at])


def view_window(m, rng, rates):
    n = len(m["bx"])
    q = rates["view_width"]
    width = _knots(n, int(rates["view_step_s"] * FPS), rng,
                   lambda r: np.interp(r.uniform(0.05, 0.95), (0.05, 0.5, 0.95), q))
    lo, hi = rates["view_ball_pos"]
    where = _knots(n, int(rates["view_step_s"] * FPS), rng, lambda r: r.uniform(lo, hi))
    ball = pd.Series(m["bx"]).ffill().bfill().fillna(0.5)
    centre = ball.rolling(2 * FPS, center=True, min_periods=1).mean().values
    left = centre - where * width
    right = left + width
    over = np.maximum(0, right - (1 + VIEW_MARGIN))           # the camera stops at the pitch ends
    left, right = left - over, right - over
    under = np.maximum(0, -VIEW_MARGIN - left)
    left, right = left + under, right + under
    px, py = m["px"].copy(), m["py"].copy()
    out_p = (px < left[:, None]) | (px > right[:, None])
    if not rates.get("view_drops_players", True):           # P9u: our tactical ATD keeps ~10.8 per team, degraded DFL ~9.7
        out_p[:] = False
    held = {}
    for k in (v for v in m["gk"].values() if v >= 0):        # the ATD keeps an unseen keeper where last seen
        seen = ~out_p[:, k] & ~np.isnan(px[:, k])
        held[k] = (pd.Series(np.where(seen, px[:, k], np.nan)).ffill().values,
                   pd.Series(np.where(seen, py[:, k], np.nan)).ffill().values)
    px[out_p], py[out_p] = np.nan, np.nan
    for k, (hx, hy) in held.items():
        px[:, k], py[:, k] = hx, hy
    bx, by = m["bx"].copy(), m["by"].copy()
    out_b = (bx < left) | (bx > right)
    bx[out_b], by[out_b] = np.nan, np.nan
    return _copy(m, px=px, py=py, bx=bx, by=by)


def drop_stationary_ball(m, rng, rates):
    L, W = m["pitch"]
    bx, by = m["bx"].copy(), m["by"].copy()
    speed = np.r_[np.nan, np.hypot(np.diff(bx) * L, np.diff(by) * W) * FPS]
    for a, b in _runs(speed < rates["still_speed"]):
        if b - a >= rates["still_frames"]:
            bx[a:b], by[a:b] = np.nan, np.nan
    for a, b in _runs((m["labels"] == pm.D) & (m["half"] > 0)):
        if rng.uniform() < rates["dead_ball_drop"]:
            bx[a:b], by[a:b] = np.nan, np.nan
    return _copy(m, bx=bx, by=by)


def lose_ball(m, rng, rates):
    """Ball losses inside the view (occlusion, tracker failure): a Poisson process over the halves,
    log-normal durations."""
    bx, by = m["bx"].copy(), m["by"].copy()
    if rates["loss_per_min"] <= 0:
        return _copy(m, bx=bx, by=by)
    idx = np.flatnonzero(m["half"] > 0)
    mean_gap = 60 * FPS / rates["loss_per_min"]
    i = int(rng.exponential(mean_gap))
    while i < len(idx):
        dur = int(round(rates["loss_median_s"] * FPS * np.exp(rates["loss_sigma"] * rng.normal())))
        sl = idx[i:i + max(1, dur)]
        bx[sl], by[sl] = np.nan, np.nan
        i += max(1, dur) + int(rng.exponential(mean_gap))
    return _copy(m, bx=bx, by=by)


def drop_gk_hold(m, rates):
    L, W = m["pitch"]
    bx, by = m["bx"].copy(), m["by"].copy()
    for t in (0, 1):
        k = m["gk"][t]
        if k < 0:
            continue
        d = np.hypot((m["px"][:, k] - m["bx"]) * L, (m["py"][:, k] - m["by"]) * W)
        for a, b in _runs(d <= rates["gk_hold_m"]):
            if b - a >= rates["gk_hold_s"] * FPS:
                bx[a:b], by[a:b] = np.nan, np.nan
    return _copy(m, bx=bx, by=by)


def ball_drift(m, rng, rates):
    L, W = m["pitch"]
    bx, by = m["bx"].copy(), m["by"].copy()
    seen = ~np.isnan(m["bx"])
    for a, b in _runs(~seen):
        if a < 2 or not seen[a - 2] or rng.uniform() >= rates["drift_p"]:
            continue
        k = np.arange(1, min(int(round(rng.uniform(*rates["drift_s"]) * FPS)), b - a) + 1)
        bx[a:a + len(k)] = m["bx"][a - 1] + (m["bx"][a - 1] - m["bx"][a - 2]) * k
        by[a:a + len(k)] = m["by"][a - 1] + (m["by"][a - 1] - m["by"][a - 2]) * k
    n, step = len(bx), FPS
    for s in range(0, n, step):
        if rng.uniform() < rates["jitter_share"]:
            r, ang = rng.uniform(*rates["jitter_m"]), rng.uniform(0, 2 * np.pi)
            sl = slice(s, s + step)
            keep = seen[sl]
            bx[sl][keep] += r * np.cos(ang) / L
            by[sl][keep] += r * np.sin(ang) / W
    return _copy(m, bx=bx, by=by)


def fragment_players(m, rng, rates, slots_per_team=23):
    n = len(m["bx"])
    team_slots = {0: list(range(1, slots_per_team)), 1: list(range(slots_per_team + 1, 2 * slots_per_team))}
    px = np.full((n, 2 * slots_per_team), np.nan, np.float32)
    py = px.copy()
    used = np.zeros((n, 2 * slots_per_team), bool)
    gk = {}
    for t in (0, 1):
        k = m["gk"][t]
        gk[t] = t * slots_per_team if k >= 0 else -1
        if k >= 0:
            px[:, gk[t]], py[:, gk[t]] = m["px"][:, k], m["py"][:, k]
            used[:, gk[t]] = ~np.isnan(m["px"][:, k])
    mean_len = rates["piece_median_s"] * FPS / np.log(2)
    for k in range(m["px"].shape[1]):
        t = int(m["slot_team"][k])
        if k == m["gk"][t]:
            continue
        for a, b in _runs(~np.isnan(m["px"][:, k])):
            s = a
            while s < b:
                e = min(b, s + max(1, int(rng.exponential(mean_len))))
                if rng.uniform() >= rates["piece_drop"]:
                    free = [j for j in team_slots[t] if not used[s:e, j].any()]
                    if free:
                        j = free[0]
                        px[s:e, j], py[s:e, j] = m["px"][s:e, k], m["py"][s:e, k]
                        used[s:e, j] = True
                s = e
    slot_team = np.r_[np.zeros(slots_per_team), np.ones(slots_per_team)].astype(np.int8)
    return _copy(m, px=px, py=py, slot_team=slot_team, gk=gk)


def double_count(m, rng, rates):
    L, W = m["pitch"]
    px, py = m["px"].copy(), m["py"].copy()
    n = len(m["bx"])
    gks = set(v for v in m["gk"].values() if v >= 0)
    s = 0
    while s < n:
        e = min(n, s + rng.randint(FPS, 2 * FPS + 1))
        if rng.uniform() < rates["dup_share"]:
            src = [k for k in np.flatnonzero(~np.isnan(m["px"][s])) if k not in gks]
            if src:
                k = src[rng.randint(len(src))]
                t = m["slot_team"][k]
                free = [j for j in np.flatnonzero(m["slot_team"] == t) if np.isnan(px[s:e, j]).all()]
                if free:
                    r, ang = rng.uniform(*rates["dup_m"]), rng.uniform(0, 2 * np.pi)
                    j = free[0]
                    px[s:e, j] = m["px"][s:e, k] + r * np.cos(ang) / L
                    py[s:e, j] = m["py"][s:e, k] + r * np.sin(ang) / W
        s = e
    return _copy(m, px=px, py=py)


def freeze_keepers(m, rng, rates):
    px, py = m["px"].copy(), m["py"].copy()
    n = len(m["bx"])
    share, mean = rates["gk_freeze_share"], rates["gk_freeze_mean_s"] * FPS
    gap_mean = mean * (1 - share) / share if share > 0 else np.inf
    for t in (0, 1):
        k = m["gk"][t]
        if k < 0 or share <= 0:
            continue
        s = int(rng.exponential(gap_mean)) if np.isfinite(gap_mean) and gap_mean > 0 else 0
        while s < n:
            e = min(n, s + max(1, int(rng.exponential(mean))))
            if not np.isnan(px[s, k]):
                px[s:e, k], py[s:e, k] = px[s, k], py[s, k]
            s = e + (int(rng.exponential(gap_mean)) if gap_mean > 0 else 0)
    return _copy(m, px=px, py=py)


def smooth_ball(m, rates):
    """The provider smooths its ball: a centred moving average of smooth_frames, inside each observed
    stretch only (the window shrinks at the stretch edges; nothing is carried across a gap)."""
    w = int(rates.get("smooth_frames", 0))
    if w <= 1:
        return m
    h = w // 2
    out = {}
    for k in ("bx", "by"):
        x = m[k].astype(np.float64)
        y = x.copy()
        for a, b in _runs(~np.isnan(x)):
            c = np.r_[0.0, np.cumsum(x[a:b])]
            i = np.arange(b - a)
            lo, hi = np.maximum(0, i - h), np.minimum(b - a, i + h + 1)
            y[a:b] = (c[hi] - c[lo]) / (hi - lo)
        out[k] = y.astype(np.float32)
    return _copy(m, **out)


def carry_dead_ball(m, rng, rates):
    """With probability dead_carry_p per dead stretch, the ball is tracked through it at the player who
    is nearest to where play restarts (the ball's position on the first frame after the stretch)."""
    p = rates.get("dead_carry_p", 0.0)
    if p <= 0:
        return m
    L, W = m["pitch"]
    bx, by = m["bx"].copy(), m["by"].copy()
    for a, b in _runs((m["labels"] == pm.D) & (m["half"] > 0)):
        if b >= len(bx) or rng.uniform() >= p:
            continue
        rx, ry = m["bx"][b], m["by"][b]
        if np.isnan(rx):
            continue
        d = np.hypot((m["px"][b - 1] - rx) * L, (m["py"][b - 1] - ry) * W)
        if np.all(np.isnan(d)):
            continue
        k = int(np.nanargmin(d))
        bx[a:b], by[a:b] = m["px"][a:b, k], m["py"][a:b, k]
    return _copy(m, bx=bx, by=by)


def degrade(m, rates=DEFAULT_RATES, seed=0):
    rng = np.random.RandomState(seed)
    out = view_window(m, rng, rates)
    out = drop_stationary_ball(out, rng, rates)
    out = carry_dead_ball(out, np.random.RandomState(seed + 2000), rates)
    out = lose_ball(out, rng, rates)
    out = drop_gk_hold(out, rates)
    out = ball_drift(out, rng, rates)
    out = smooth_ball(out, rates)
    out = fragment_players(out, rng, rates)
    out = double_count(out, rng, rates)
    out = freeze_keepers(out, rng, rates)
    out["name"] = m["name"] + "+deg"
    return out


# A video ball detector next to the ATD ball, at the rates measured for ball_v1 + online tracking on
# our match's validation blocks (analysis/p6a_ball_tracking.py, notes/BULGULAR.md §10). Not part of
# degrade(): earlier numbers stay as they are.
DETECTOR_RATES = {
    "report": 0.6,              # share of play with a detection: 62.5% coverage, 58% of in-play ATD gaps
    "good": 0.68,               # of those, within 1.5 m of the ball (precision against the ATD)
    "noise_m": 0.5,             # position noise of a good detection
    "chunk_s": 1.0,             # detections come and go in stretches (mean length)
    "wrong_near_player": 0.5,   # a wrong detection sits on a player (a head, a boot), else off by 2-10 m
}


def add_detector(clean, deg, seed=0, rates=DETECTOR_RATES):
    """deg with dx, dy: a simulated detector track of the clean ball, NaN when nothing is reported."""
    rng = np.random.RandomState(seed + 1000)
    L, W = deg["pitch"]
    n = len(clean["bx"])
    dx, dy = np.full(n, np.nan), np.full(n, np.nan)
    t = 0
    while t < n:
        k = max(1, int(rng.exponential(rates["chunk_s"] * 25)))
        s = slice(t, min(n, t + k))
        t += k
        if rng.rand() >= rates["report"]:
            continue
        bx, by = clean["bx"][s].astype(float), clean["by"][s].astype(float)
        if rng.rand() < rates["good"]:
            dx[s] = bx + rng.normal(0, rates["noise_m"], len(bx)) / L
            dy[s] = by + rng.normal(0, rates["noise_m"], len(by)) / W
        elif rng.rand() < rates["wrong_near_player"]:
            f = s.start
            cand = np.flatnonzero(~np.isnan(deg["px"][f]))
            if not len(cand):
                continue
            j = cand[rng.randint(len(cand))]
            dx[s], dy[s] = deg["px"][s, j], deg["py"][s, j]
        else:
            a, r = rng.uniform(0, 2 * np.pi), rng.uniform(2, 10)
            dx[s] = bx + r * np.cos(a) / L
            dy[s] = by + r * np.sin(a) / W
    off = (deg["half"] == 0) | np.isnan(clean["bx"])
    dx[off], dy[off] = np.nan, np.nan
    out = dict(deg)
    out.update(dx=dx.astype(np.float32), dy=dy.astype(np.float32))
    return out
