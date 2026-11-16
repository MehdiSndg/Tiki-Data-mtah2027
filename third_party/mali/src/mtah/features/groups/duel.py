"""dl (Gün 5, rapor 6 §IMPLEMENTATION BRIEF 2; Ek M.12): duel dipole and point-wise pitch control.

Frame-level (no touch events needed: on BAR-ATH kinematic touch events bracket only ~1/3 of the
true changes, Ek M.12). Source-common, permutation-invariant (nearest player per team only),
home perspective (positive = home). NaN where the ball is hidden. Distances are robust z-scores
over the match's visible frames, so BAR's smoothed / offset ball (nearest player 0.90 m vs DFL
0.63 m with a slow ball) and DFL share the zero point.

Columns
  dl_dist_z_H / dl_dist_z_A   z(min over [t-4, t+2] of the team's nearest-player distance)
  dl_dist_z_diff              dl_dist_z_A - dl_dist_z_H (> 0: home closer)
  dl_close_diff               z(closing speed home) - z(closing speed away); closing = -d/dt dist (5 frames)
  dl_out_align_diff           cos(ball out-velocity, home nearest player's velocity at t-2) - same for away;
                              out-velocity = (b(t+6) - b(t+1)) / 0.2 s
  dl_deflect                  angle between in-velocity (b(t-1) - b(t-6)) and out-velocity (rad)
  dl_shield                   body-shielding cosine of the home nearest player minus the away one's:
                              cos((pH - pA), (b - pA)) - cos((pA - pH), (b - pH))
  dl_tti_now                  2 sigmoid(pi / sqrt(3) / 0.45 * dtau) - 1, dtau = min_away tau - min_home tau,
                              tau = t_react + |r - (p + v t_react)| / v_max at r = ball (Spearman 2018)
  dl_ppcf_1s                  the same at r = b + 1.0 s * out-velocity (where the ball is going)
  dl_ppcf_gain                dl_ppcf_1s(t+3) - dl_ppcf_1s(t-3)
  dl_event_team               -0.5 (dist_z_H - dist_z_A) + 0.25 out_align_diff + 0.25 ppcf_1s (continuous
                              version of rapor 6's kinematic_event_team)
"""

import numpy as np

from ..context import nan_call
from ..registry import register

DEFAULTS = dict(t_react=0.5, v_max=7.0, sigma_t=0.45, horizon_s=1.0)


def _z(x, ref):
    v = x[ref & np.isfinite(x)]
    if len(v) < 10:
        return x * np.nan
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    return np.clip((x - med) / max(mad, 1e-6), -5.0, 5.0)


def _shift(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def _tti_margin(P, V, home, r, kc):
    """2 sigmoid(c * (min_away tau - min_home tau)) - 1 at points r (T, 2)."""
    drift = P + V * kc["t_react"]
    tau = kc["t_react"] + np.linalg.norm(r[:, None, :] - drift, axis=-1) / kc["v_max"]
    th = nan_call(np.nanmin, np.where(home[None], tau, np.nan), axis=1)
    ta = nan_call(np.nanmin, np.where(~home[None], tau, np.nan), axis=1)
    c = np.pi / np.sqrt(3) / kc["sigma_t"]
    return 2.0 / (1.0 + np.exp(-np.clip(c * (ta - th), -30, 30))) - 1.0


@register("dl", context=["dl_ppcf_1s", "dl_dist_z_diff"])
def duel(ctx):
    kc = dict(DEFAULTS, **ctx.cfg["features"].get("duel", {}))
    n, fps = ctx.n, ctx.fps
    B = ctx.B.astype(np.float64)
    P = ctx.P.astype(np.float64)
    vis = ctx.vis
    playing = ctx.data.period > 0
    ref = vis & playing
    home = ctx.home
    V = np.gradient(P, 1.0 / fps, axis=0)

    dist = np.linalg.norm(P - B[:, None, :], axis=-1)
    dist = np.where(np.isnan(dist), np.inf, dist)
    jh = np.argmin(np.where(home[None], dist, np.inf), axis=1)
    ja = np.argmin(np.where(~home[None], dist, np.inf), axis=1)
    rows = np.arange(n)
    dh = np.where(np.isfinite(dist[rows, jh]), dist[rows, jh], np.nan)
    da = np.where(np.isfinite(dist[rows, ja]), dist[rows, ja], np.nan)

    def win_min(x):
        stack = np.stack([_shift(x, k) for k in range(-4, 3)])
        return nan_call(np.nanmin, stack, axis=0)

    zh, za = _z(win_min(dh), ref), _z(win_min(da), ref)
    close_h = -(dh - _shift(dh, -5)) / (5 / fps)
    close_a = -(da - _shift(da, -5)) / (5 / fps)
    close_diff = _z(close_h, ref) - _z(close_a, ref)

    v_out = (_shift(B, 6) - _shift(B, 1)) / (5 / fps)
    v_in = (_shift(B, -1) - _shift(B, -6)) / (5 / fps)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosd = np.sum(v_in * v_out, 1) / (np.linalg.norm(v_in, axis=1) * np.linalg.norm(v_out, axis=1) + 1e-5)
    deflect = np.arccos(np.clip(cosd, -1, 1))

    back = np.clip(rows - 2, 0, n - 1)
    vph = V[back, jh]                     # the CURRENT nearest player's velocity 2 frames ago
    vpa = V[back, ja]

    def cosv(a, b):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-5)

    align_diff = cosv(v_out, vph) - cosv(v_out, vpa)
    ph, pa = P[rows, jh], P[rows, ja]
    shield = cosv(ph - pa, B - pa) - cosv(pa - ph, B - ph)

    tti_now = _tti_margin(P, V, home, B, kc)
    r1 = B + kc["horizon_s"] * v_out
    ppcf = _tti_margin(P, V, home, r1, kc)
    gain = _shift(ppcf, 3) - _shift(ppcf, -3)
    ev_team = -0.5 * (zh - za) + 0.25 * align_diff + 0.25 * ppcf

    hide = ~vis
    out = {"dl_dist_z_H": zh, "dl_dist_z_A": za, "dl_dist_z_diff": za - zh, "dl_close_diff": close_diff,
           "dl_out_align_diff": align_diff, "dl_deflect": deflect, "dl_shield": shield, "dl_tti_now": tti_now,
           "dl_ppcf_1s": ppcf, "dl_ppcf_gain": gain, "dl_event_team": ev_team}
    return {k: np.where(hide, np.nan, v) for k, v in out.items()}
