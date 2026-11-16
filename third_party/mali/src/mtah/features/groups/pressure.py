"""press (Group 3): pressure and space control around the action.

Focal point: the ball when visible, otherwise the "contest point" (midpoint of the closest
home-away outfield pair), so every feature exists through ball gaps.
  * local density per team (Gaussian, sigma 5 m) and head counts within 5 / 10 m
  * closing speed towards the focal point (sum of positive radial velocity, players within 15 m)
  * time-to-intercept (Spearman-style: reaction 0.7 s then straight run at v_max) at the focal
    point and at a 0.4 s ball look-ahead point; softmin pitch control P(home)
  * passing lanes: teammates 3-35 m from the ball whose straight lane has no opponent within
    1.5 m (ball visible only); the share of open lanes per team
"""

import numpy as np

from ..context import nan_call
from ..registry import register

DEFAULTS = dict(sigma_m=5.0, close_radius_m=15.0, react_s=0.7, vmax_ms=6.0, lam=4.3,
                lookahead_s=0.4, lane_block_m=1.5, lane_min_m=3.0, lane_max_m=35.0, chunk=8000)


def _contest_point(P, home_of, away_of, chunk):
    n = len(P)
    out = np.full((n, 2), np.nan, np.float32)
    ph_all = np.where(home_of[..., None], P, np.nan)
    pa_all = np.where(away_of[..., None], P, np.nan)
    for s in range(0, n, chunk):
        ph, pa = ph_all[s:s + chunk], pa_all[s:s + chunk]
        d = np.hypot(ph[:, :, None, 0] - pa[:, None, :, 0], ph[:, :, None, 1] - pa[:, None, :, 1])
        flat = np.where(np.isnan(d), np.inf, d).reshape(len(d), -1)
        k = np.argmin(flat, axis=1)
        ok = np.isfinite(flat[np.arange(len(d)), k])
        i, j = np.unravel_index(k, d.shape[1:])
        mid = (ph[np.arange(len(d)), i] + pa[np.arange(len(d)), j]) / 2
        out[s:s + chunk] = np.where(ok[:, None], mid, np.nan)
    return out


def _tti(P, V, r, react, vmax):
    """(T,N) time for each player to reach point r (T,2)."""
    ahead = P + V * react
    return react + np.hypot(ahead[..., 0] - r[:, None, 0], ahead[..., 1] - r[:, None, 1]) / vmax


def _softmin_share(tti, home, away, lam):
    w = np.exp(-lam * np.where(np.isnan(tti), np.inf, tti))
    wh = np.sum(np.where(home, w, 0.0), axis=1)
    wa = np.sum(np.where(away, w, 0.0), axis=1)
    tot = wh + wa
    return np.where(tot > 0, wh / np.where(tot > 0, tot, 1), np.nan)


def _lanes(B, P, mate_mask, opp_mask, cfg):
    """Open passing lanes from the ball to each team's players (count, share)."""
    n = len(B)
    count = np.full(n, np.nan, np.float32)
    share = np.full(n, np.nan, np.float32)
    mates_all = np.where(mate_mask[..., None], P, np.nan)
    opps_all = np.where(opp_mask[..., None], P, np.nan)
    for s in range(0, n, cfg["chunk"]):
        b = B[s:s + cfg["chunk"]].astype(np.float64)
        m = mates_all[s:s + cfg["chunk"]] - b[:, None, :]
        o = opps_all[s:s + cfg["chunk"]] - b[:, None, :]
        dm = np.hypot(m[..., 0], m[..., 1])
        cand = (dm >= cfg["lane_min_m"]) & (dm <= cfg["lane_max_m"])
        dot = m[:, :, None, 0] * o[:, None, :, 0] + m[:, :, None, 1] * o[:, None, :, 1]
        u = dot / np.maximum(dm[:, :, None] ** 2, 1e-6)
        px, py = u * m[:, :, None, 0], u * m[:, :, None, 1]
        d = np.hypot(o[:, None, :, 0] - px, o[:, None, :, 1] - py)
        blocked = np.any((u > 0.1) & (u < 0.95) & (d < cfg["lane_block_m"]), axis=2)
        n_cand = cand.sum(axis=1)
        n_open = (cand & ~blocked).sum(axis=1)
        vis = ~np.isnan(b[:, 0])
        count[s:s + cfg["chunk"]] = np.where(vis, n_open, np.nan)
        share[s:s + cfg["chunk"]] = np.where(vis & (n_cand > 0), n_open / np.maximum(n_cand, 1), np.nan)
    return count, share


@register("press", context=["press_dens_diff", "press_close_diff", "press_pc_focal",
                            "press_lanes_diff"])
def pressure(ctx):
    pc = dict(DEFAULTS, **ctx.cfg["features"].get("pressure", {}))
    P, V, B = ctx.P.astype(np.float64), ctx.vp.astype(np.float64), ctx.B.astype(np.float64)
    home, away = ctx.home, ctx.away
    of_h, of_a = ctx.outfield_home, ctx.outfield_away

    contest = _contest_point(ctx.P, of_h, of_a, int(pc["chunk"])).astype(np.float64)
    focal = np.where(ctx.vis[:, None], B, contest)

    dx, dy = focal[:, None, 0] - P[..., 0], focal[:, None, 1] - P[..., 1]
    d = np.hypot(dx, dy)
    g = np.exp(-(d ** 2) / (2 * pc["sigma_m"] ** 2))
    out = {}
    for tag, mask in (("H", home), ("A", away)):
        out[f"press_dens_{tag}"] = np.nansum(np.where(mask, g, 0.0), axis=1)
        out[f"press_n5_{tag}"] = np.sum(mask & (d < 5), axis=1)
        out[f"press_n10_{tag}"] = np.sum(mask & (d < 10), axis=1)
        radial = (dx * V[..., 0] + dy * V[..., 1]) / np.maximum(d, 0.5)
        close = np.where(mask & (d < pc["close_radius_m"]), np.maximum(radial, 0.0), 0.0)
        out[f"press_close_{tag}"] = np.nansum(close, axis=1)
    out["press_dens_diff"] = out["press_dens_H"] - out["press_dens_A"]
    out["press_close_diff"] = out["press_close_H"] - out["press_close_A"]
    out["press_focal_is_ball"] = ctx.vis.astype(np.float32)

    tti_f = _tti(P, V, focal, pc["react_s"], pc["vmax_ms"])
    out["press_pc_focal"] = _softmin_share(tti_f, home, away, pc["lam"])
    tmin_h = nan_call(np.nanmin, np.where(home, tti_f, np.nan), axis=1)
    tmin_a = nan_call(np.nanmin, np.where(away, tti_f, np.nan), axis=1)
    out["press_tti_diff_focal"] = tmin_a - tmin_h

    look = B + ctx.vb.astype(np.float64) * pc["lookahead_s"]
    tti_l = _tti(P, V, look, pc["react_s"], pc["vmax_ms"])
    out["press_pc_look"] = _softmin_share(tti_l, home, away, pc["lam"])
    out["press_tti_diff_look"] = (nan_call(np.nanmin, np.where(away, tti_l, np.nan), axis=1)
                                  - nan_call(np.nanmin, np.where(home, tti_l, np.nan), axis=1))

    cnt_h, sh_h = _lanes(B, ctx.P, home, away, pc)
    cnt_a, sh_a = _lanes(B, ctx.P, away, home, pc)
    out.update(press_lanes_H=cnt_h, press_lanes_A=cnt_a, press_lanes_diff=cnt_h - cnt_a,
               press_lane_share_H=sh_h, press_lane_share_A=sh_a)
    return out
