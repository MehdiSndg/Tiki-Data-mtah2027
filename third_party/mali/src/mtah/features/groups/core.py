"""core (M0): ball state, dead-ball geometry, proximity, tempo, basic team shape, gap edges."""

import numpy as np
import pandas as pd

from ..context import nan_call
from ..registry import register

SIX_YARD_HALF_WIDTH = 9.16
SIX_YARD_DEPTH = 5.5


def _rect_dist(x, y, x0, x1, y0, y1):
    dx = np.maximum(np.maximum(x0 - x, 0), x - x1)
    dy = np.maximum(np.maximum(y0 - y, 0), y - y1)
    return np.hypot(dx, dy)


def gap_features(vis, values, fps):
    """Gap length/position and `values` at the last / next visible frame around each gap."""
    n = len(vis)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(vis, idx, -1))
    nxt = np.minimum.accumulate(np.where(vis, idx, n)[::-1])[::-1]
    gap_len = np.where(vis, 0, np.where((last >= 0) & (nxt < n), nxt - last - 1, n))
    since = np.where(vis, 0, idx - last).astype(np.float32)
    until = np.where(vis, 0, nxt - idx).astype(np.float32)
    since[last < 0] = np.nan
    until[nxt >= n] = np.nan
    out = {"gap_len_s": gap_len / fps, "gap_since_s": since / fps, "gap_until_s": until / fps,
           "gap_pos": np.where(vis, np.nan, since / np.maximum(1, gap_len))}
    for name, v in values.items():
        out[f"gap_{name}_prev"] = np.where(last >= 0, v[np.clip(last, 0, n - 1)], np.nan)
        out[f"gap_{name}_next"] = np.where(nxt < n, v[np.clip(nxt, 0, n - 1)], np.nan)
    return out


@register("core", context=["prox_ddiff", "prox_contact", "pl_p80_speed", "ball_oob_flag",
                           "team_adv_diff", "team_cen_att_diff", "ball_vis", "ball_speed",
                           "prox_dmin"])
def core(ctx):
    fc = ctx.cfg["features"]
    L, W = ctx.length, ctx.width
    bx, by = ctx.B[:, 0], ctx.B[:, 1]
    vis = ctx.vis
    f = {}
    f["ball_vis"] = vis.astype(np.float32)
    f["ball_xc"] = ctx.xc_b
    f["ball_y"] = by
    f["ball_speed"] = ctx.sb

    oob = np.maximum.reduce([-bx, bx - L, -by, by - W])
    f["ball_oob_dist"] = oob
    corners = [(0, 0), (0, W), (L, 0), (L, W)]
    f["ball_d_corner"] = np.min([np.hypot(bx - cx, by - cy) for cx, cy in corners], axis=0)
    f["ball_d_center"] = np.hypot(bx - L / 2, by - W / 2)
    y0, y1 = W / 2 - SIX_YARD_HALF_WIDTH, W / 2 + SIX_YARD_HALF_WIDTH
    f["ball_d_goalarea"] = np.minimum(_rect_dist(bx, by, 0, SIX_YARD_DEPTH, y0, y1),
                                      _rect_dist(bx, by, L - SIX_YARD_DEPTH, L, y0, y1))
    f["ball_d_touchline"] = np.minimum(np.abs(by), np.abs(W - by))
    f["ball_oob_flag"] = np.where(vis, (oob > fc["oob_tolerance_m"]).astype(np.float32), np.nan)

    d_h, d_a = ctx.d_home, ctx.d_away
    f["prox_dH"], f["prox_dA"] = d_h, d_a
    f["prox_ddiff"] = d_a - d_h
    f["prox_dmin"] = np.fmin(d_h, d_a)
    f["prox_nH_r3"] = np.where(vis, np.sum((ctx.dist < 3) & ctx.home, axis=1), np.nan)
    f["prox_nA_r3"] = np.where(vis, np.sum((ctx.dist < 3) & ctx.away, axis=1), np.nan)
    r = fc["contact_radius_m"]
    f["prox_contact"] = np.where(vis, (d_h < r).astype(np.float32) - (d_a < r), np.nan)

    sp_out = np.where(ctx.outfield, ctx.sp, np.nan)
    f["pl_p80_speed"] = nan_call(np.nanpercentile, sp_out, 80, axis=1)
    f["pl_mean_speed"] = nan_call(np.nanmean, sp_out, axis=1)
    f["pl_n_home"] = np.sum(~np.isnan(ctx.P[:, ctx.home, 0]), axis=1)
    f["pl_n_away"] = np.sum(~np.isnan(ctx.P[:, ctx.away, 0]), axis=1)

    of_h, of_a = ctx.outfield_home, ctx.outfield_away
    cen_h = nan_call(np.nanmean, np.where(of_h, ctx.xc_p, np.nan), axis=1)
    cen_a = nan_call(np.nanmean, np.where(of_a, L - ctx.xc_p, np.nan), axis=1)
    f["team_cen_att_H"], f["team_cen_att_A"] = cen_h, cen_a
    f["team_cen_att_diff"] = cen_h - cen_a
    adv_h = nan_call(np.nanmean, np.where(of_h, ctx.vx_c, np.nan), axis=1)
    adv_a = nan_call(np.nanmean, np.where(of_a, -ctx.vx_c, np.nan), axis=1)
    f["team_adv_H"], f["team_adv_A"] = adv_h, adv_a
    f["team_adv_diff"] = adv_h - adv_a
    sy_h = nan_call(np.nanstd, np.where(of_h, ctx.y_p, np.nan), axis=1)
    sy_a = nan_call(np.nanstd, np.where(of_a, ctx.y_p, np.nan), axis=1)
    sx_h = nan_call(np.nanstd, np.where(of_h, ctx.xc_p, np.nan), axis=1)
    sx_a = nan_call(np.nanstd, np.where(of_a, ctx.xc_p, np.nan), axis=1)
    f["team_spread_y_diff"], f["team_spread_x_diff"] = sy_h - sy_a, sx_h - sx_a
    xs_h = np.sort(np.where(of_h, ctx.xc_p, np.nan), axis=1)
    xs_a = np.sort(np.where(of_a, L - ctx.xc_p, np.nan), axis=1)
    f["team_defline_H"], f["team_defline_A"] = xs_h[:, 1], xs_a[:, 1]

    team_sig = pd.Series(ctx.team_sign_near).rolling(11, center=True, min_periods=1).mean()
    f.update(gap_features(vis, {"team": team_sig.to_numpy(), "dmin": f["prox_dmin"], "oob": oob,
                                "speed": np.nan_to_num(ctx.sb, nan=0.0)}, ctx.fps))
    return f
