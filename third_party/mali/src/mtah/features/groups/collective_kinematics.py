"""ck (Gün 4, rapor 5 §2.1 / Implementation Brief §3): ball-free collective kinematics.

Source-common (DFL / Metrica have the same inputs), slot-permutation invariant, outfield players
only (goalkeepers from preprocess.goalkeepers_and_direction, never fixed slot numbers). Common
frame = home attacks +x (ctx.vx_c). Targets error F3: truly alive, slow (p80 ~1.6 m/s), hidden-
ball frames the model calls dead; the scalar tempo features cannot tell slow circulation from a
stoppage, the DIRECTION structure of the movement can.

Columns
  ck_adv_signed      med_H(vx_c) + med_A(vx_c): both teams' x-velocity in the home-attack direction
                     (= rapor 5 'adv_vel_diff_signed' and 2 x 'team_flow_attack_dir'; mean version
                     is core.team_adv_diff)
  ck_adv_mag         |ck_adv_signed|  (coupled shift of both blocks; ~0 in a stoppage)
  ck_phi_H/A, ck_phi_max   polar order |mean_i v_i / (|v_i| + 0.1)| per team (direction agreement)
  ck_cen_vel_cos     cosine between the two teams' mean velocity vectors
  ck_width_logratio  ln((y_p90 - y_p10)_H + 1) / ((y_p90 - y_p10)_A + 1)
  ck_ball_exit       E_out: ball hidden and its last visible position, projected 1 s along its last
                     velocity, is outside / within 2 m of a pitch line (sigmoid, 0 when visible)
  ck_missing_s       seconds since the ball was last visible (0 when visible)
  ck_missing_coherent  ln(1 + missing_s) * (1 - E_out) * sigmoid(ck_adv_mag - 1.0)
  ck_last_owner      sign of the team nearest to the ball when it was last seen, decayed exp(-t/12 s)
  ck_restart_spot    an outfield player within 1.5 m of a pitch line and slower than 0.6 m/s
  ck_gk_depth_H/A    goalkeeper distance from its own goal line (m)
  ck_axis_align      |e1_H . e1_A| * min(anisotropy_H, anisotropy_A) of the teams' position PCA
"""

import numpy as np
import pandas as pd

from ..context import nan_call
from ..registry import register


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _polar(vx, vy, mask):
    s = np.hypot(vx, vy) + 0.1
    ux = np.where(mask, vx / s, np.nan)
    uy = np.where(mask, vy / s, np.nan)
    return np.hypot(nan_call(np.nanmean, ux, axis=1), nan_call(np.nanmean, uy, axis=1))


def _pca(x, y, mask):
    """(principal axis unit vector (T,2), anisotropy lambda1 / (lambda1 + lambda2))."""
    xx = np.where(mask, x, np.nan)
    yy = np.where(mask, y, np.nan)
    mx, my = nan_call(np.nanmean, xx, axis=1), nan_call(np.nanmean, yy, axis=1)
    dx, dy = xx - mx[:, None], yy - my[:, None]
    sxx = nan_call(np.nanmean, dx * dx, axis=1)
    syy = nan_call(np.nanmean, dy * dy, axis=1)
    sxy = nan_call(np.nanmean, dx * dy, axis=1)
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    ex, ey = sxy, l1 - sxx                                 # eigenvector of l1
    alt = np.abs(sxy) < 1e-9
    ex = np.where(alt, np.where(sxx >= syy, 1.0, 0.0), ex)
    ey = np.where(alt, np.where(sxx >= syy, 0.0, 1.0), ey)
    n = np.hypot(ex, ey) + 1e-12
    return np.stack([ex / n, ey / n], axis=1), l1 / np.maximum(l1 + l2, 1e-9)


@register("ck", context=["ck_adv_mag", "ck_phi_max", "ck_cen_vel_cos", "ck_adv_signed"])
def collective_kinematics(ctx):
    L, W, fps = ctx.length, ctx.width, ctx.fps
    of_h, of_a = ctx.outfield_home, ctx.outfield_away
    vx_c, vy = ctx.vx_c, ctx.vp[..., 1]
    f = {}
    med_h = nan_call(np.nanmedian, np.where(of_h, vx_c, np.nan), axis=1)
    med_a = nan_call(np.nanmedian, np.where(of_a, vx_c, np.nan), axis=1)
    adv = med_h + med_a
    f["ck_adv_signed"] = adv
    f["ck_adv_mag"] = np.abs(adv)
    vx = ctx.vp[..., 0]
    phi_h, phi_a = _polar(vx, vy, of_h), _polar(vx, vy, of_a)
    f["ck_phi_H"], f["ck_phi_A"], f["ck_phi_max"] = phi_h, phi_a, np.fmax(phi_h, phi_a)
    mvx_h = nan_call(np.nanmean, np.where(of_h, vx, np.nan), axis=1)
    mvy_h = nan_call(np.nanmean, np.where(of_h, vy, np.nan), axis=1)
    mvx_a = nan_call(np.nanmean, np.where(of_a, vx, np.nan), axis=1)
    mvy_a = nan_call(np.nanmean, np.where(of_a, vy, np.nan), axis=1)
    f["ck_cen_vel_cos"] = (mvx_h * mvx_a + mvy_h * mvy_a) / (np.hypot(mvx_h, mvy_h) * np.hypot(mvx_a, mvy_a) + 1e-3)
    y = ctx.y_p
    q_h = nan_call(np.nanpercentile, np.where(of_h, y, np.nan), [10, 90], axis=1)
    q_a = nan_call(np.nanpercentile, np.where(of_a, y, np.nan), [10, 90], axis=1)
    f["ck_width_logratio"] = np.log((q_h[1] - q_h[0] + 1.0) / (q_a[1] - q_a[0] + 1.0))

    # ball-gap memory: last visible position / velocity, projected 1 s ahead
    vis = ctx.vis
    bx = pd.Series(np.where(vis, ctx.B[:, 0], np.nan)).ffill().to_numpy()
    by = pd.Series(np.where(vis, ctx.B[:, 1], np.nan)).ffill().to_numpy()
    vbx = pd.Series(np.where(vis, np.nan_to_num(ctx.vb[:, 0]), np.nan)).ffill().fillna(0).to_numpy()
    vby = pd.Series(np.where(vis, np.nan_to_num(ctx.vb[:, 1]), np.nan)).ffill().fillna(0).to_numpy()
    px, py = bx + vbx, by + vby
    d_edge = np.minimum.reduce([px, L - px, py, W - py])           # < 0 = projected outside
    e_out = np.where(vis, 0.0, _sig((2.0 - d_edge) / 0.8))
    e_out = np.where(np.isnan(bx), 0.0, e_out)
    f["ck_ball_exit"] = e_out
    idx = np.arange(ctx.n)
    last = pd.Series(np.where(vis, idx, np.nan)).ffill().to_numpy()
    missing = np.where(np.isnan(last), 0.0, (idx - np.nan_to_num(last)) / fps)
    f["ck_missing_s"] = missing
    f["ck_missing_coherent"] = np.log1p(missing) * (1 - e_out) * _sig(np.nan_to_num(f["ck_adv_mag"]) - 1.0)
    owner = pd.Series(np.where(vis, ctx.team_sign_near, np.nan)).ffill().fillna(0).to_numpy()
    f["ck_last_owner"] = owner * np.exp(-missing / 12.0)

    # set-piece cues
    sp = ctx.sp
    d_line = np.minimum.reduce([ctx.P[..., 0], L - ctx.P[..., 0], ctx.P[..., 1], W - ctx.P[..., 1]])
    near_slow = ctx.outfield & (np.nan_to_num(d_line, nan=99) < 1.5) & (np.nan_to_num(sp, nan=99) < 0.6)
    f["ck_restart_spot"] = near_slow.any(axis=1).astype(np.float32)
    gk_h = np.full(ctx.n, np.nan)
    gk_a = np.full(ctx.n, np.nan)
    from ...preprocess import filled_period
    phase = filled_period(ctx.data.period)
    for per, (gh, ga, _) in ctx.gk_dir.items():
        sel = phase == per
        gk_h[sel] = ctx.xc_p[sel, gh]
        gk_a[sel] = L - ctx.xc_p[sel, ga]
    f["ck_gk_depth_H"], f["ck_gk_depth_A"] = gk_h, gk_a
    e_h, r_h = _pca(ctx.xc_p, y, of_h)
    e_a, r_a = _pca(ctx.xc_p, y, of_a)
    f["ck_axis_align"] = np.abs(np.sum(e_h * e_a, axis=1)) * np.fmin(r_h, r_a)
    return f
