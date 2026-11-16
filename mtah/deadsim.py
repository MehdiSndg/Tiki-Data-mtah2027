"""A dead ball that looks like a tracked dead ball (P9dv; the friend's V6 point, Enes's _dead_ball_carry for the path).

Our degradation shows the dead ball almost never in the seconds before a restart and switches it on exactly at the
restart (P9dt: P(seen) 0.00-0.13 one second before, 0.87-0.91 at the restart), so a model can learn "the ball
reappears -> play restarts". Real tracking does not do that (label-free, our ATD around the video head's restarts:
0.40-0.54 before; PFF World Cup broadcast tracking around its restarts: 0.47-0.49 before, 0.65 at the restart), and
after a stoppage the ball fades out over ~2 s instead of vanishing at once (ATD 0.92 -> 0.41, PFF 0.95 -> 0.34).

realistic_dead(deg, raw, seed) rewrites the ball inside every dead stretch of a degraded copy:
  * first ~0.5-2.5 s after the stoppage the ball keeps its last live velocity, decaying (seen);
  * then it is carried by the player nearest the restart spot with a slowly drifting offset: it starts at N(0, OFF_M =
    2.5 m) per axis and follows an Ornstein-Uhlenbeck step (decay 0.002 per frame, i.e. a 20 s time constant, noise
    0.04 m) whose stationary sd is OU_SD_M ~0.63 m, so on stoppages longer than ~20 s the ball drifts in towards the
    carrier; it converges to the restart spot over the last second;
  * it is seen in on / off chunks (on: exponential, mean ON_S) whose off-length is drawn from the on-share at the chunk's
    start: NEAR_P within NEAR_S seconds of the restart, FAR_P before that. Long off chunks started earlier run into the
    last NEAR_S seconds, so the realised share there is lower than NEAR_P (14 Nov test: ~0.25 near vs ~0.14 before on a
    320 s stretch); the calibration (P9dt: 0.36-0.42 before restarts on J03WMX, our ATD 0.40-0.54) was measured on this
    realised behaviour, and every P9dv+ result used it, so it is kept as is.
Live frames are untouched. Labels are only used for where the dead stretches are (external data only).
"""

import numpy as np

FPS = 25
ON_S = 3.0
NEAR_S = 12.0
NEAR_P = 0.5
FAR_P = 0.15
OFF_M = 2.5
OU_DECAY, OU_NOISE_M = 0.002, 0.04
OU_SD_M = OU_NOISE_M / np.sqrt(1 - (1 - OU_DECAY) ** 2)       # ~0.63 m


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _onoff(n, p_on, rng):
    """Alternating renewal process with on-share p_on (per frame array or scalar)."""
    out = np.zeros(n, bool)
    t = 0
    on = rng.uniform() < np.mean(p_on)
    while t < n:
        p = float(np.clip(p_on[t] if np.ndim(p_on) else p_on, 0.02, 0.98))
        mean = ON_S if on else ON_S * (1 - p) / p
        ln = max(1, int(rng.exponential(mean * FPS)))
        if on:
            out[t:t + ln] = True
        t += ln
        on = not on
    return out


def realistic_dead(deg, raw, seed, near_p=NEAR_P, far_p=FAR_P):
    """Copy of the degraded match `deg` with the dead ball rewritten; `raw` is the clean match (restart spots)."""
    import possession_metrics as pm
    rng = np.random.default_rng(seed)
    L, W = deg["pitch"]
    sc = np.array([L, W])
    bx, by = deg["bx"].astype(np.float64).copy(), deg["by"].astype(np.float64).copy()
    lab, half = deg["labels"], deg["half"]
    n_all = len(bx)
    for a, b in _runs((lab == pm.D) & (half > 0)):
        n = b - a
        if b >= n_all or a == 0 or half[b] != half[a] or half[a - 1] != half[a] or n < FPS:
            continue
        rs = np.array([raw["bx"][b], raw["by"][b]], np.float64)
        if np.isnan(rs[0]):
            continue
        pos = np.full((n, 2), np.nan)
        # 1. after the stoppage: last live velocity, decaying
        k0 = min(n, int(rng.uniform(0.5, 2.5) * FPS))
        p0 = np.array([raw["bx"][a - 1], raw["by"][a - 1]], np.float64) * sc
        p1 = np.array([raw["bx"][a - 3], raw["by"][a - 3]], np.float64) * sc if a >= 3 else p0
        if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
            v = (p0 - p1) / 2.0
            tt = np.arange(1, k0 + 1)
            pos[:k0] = p0 + v[None, :] * (1 - np.exp(-tt / FPS))[:, None] * FPS
        # 2. carried by the player nearest the restart spot
        rawd = np.hypot((raw["px"][b - 1] - rs[0]) * L, (raw["py"][b - 1] - rs[1]) * W)
        if np.all(np.isnan(rawd)):
            continue
        j = int(np.nanargmin(rawd))
        pj = np.stack([raw["px"][a:b, j], raw["py"][a:b, j]], 1).astype(np.float64) * sc
        ok = np.isfinite(pj[:, 0])
        if ok.sum() < 2:
            continue
        idx = np.arange(n)
        pj[:, 0] = np.interp(idx, idx[ok], pj[ok, 0])
        pj[:, 1] = np.interp(idx, idx[ok], pj[ok, 1])
        off = np.zeros((n, 2))
        o = rng.normal(0, OFF_M, 2)
        for t in range(n):
            o += -OU_DECAY * o + rng.normal(0, OU_NOISE_M, 2)
            off[t] = o
        path = pj + off + rng.normal(0, 0.05, (n, 2))
        tail = min(n, FPS)
        w = np.linspace(0, 1, tail)[:, None]
        path[n - tail:] = (1 - w) * path[n - tail:] + w * (rs * sc)[None, :]
        pos[k0:] = path[k0:]
        # 3. seen in chunks, more often near the restart
        to_restart = (n - idx) / FPS
        p_on = np.where(to_restart <= NEAR_S, near_p, far_p)
        seen = _onoff(n, p_on, rng)
        seen[:k0] = True
        new = np.where(seen[:, None], pos, np.nan)
        bx[a:b], by[a:b] = new[:, 0] / L, new[:, 1] / W
    out = dict(deg)
    out["bx"], out["by"] = bx.astype(np.float32), by.astype(np.float32)
    return out
