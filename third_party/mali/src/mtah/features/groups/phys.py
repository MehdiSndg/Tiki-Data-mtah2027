"""phys (Gün 2): physical vetoes from report 1 (sections 2-3). Common to all sources.

Aerial bypass veto: without ball height a long pass looks like the ball touching every player
it flies over. A flight is: ball speed > 12 m/s, direction change < 5 deg and normal acceleration
< 2 m/s^2 (a straight, unforced trajectory). During a flight the "who is nearest" proximity
signal is replaced by NaN, and time-since-flight-start is exposed.

Goalkeeper holding (live) vs goal kick (dead): the keeper inside his own penalty area, ball
within 1.8 m, keeper moving 1-4 m/s (walking/jogging with the ball in hand) AND the ball has
not been out of the pitch in the last 10 s -> live possession of the keeper's team. With a
recent out-of-play the same picture is goal-kick preparation (dead).

Ball kinematics come from the kin group's code (SG segments, teleport splits): this module's
cache hash therefore includes kinematics.py too (DEPENDS).
"""

import numpy as np
import pandas as pd

from ..registry import register
from ...preprocess import filled_period
from .kinematics import kinematics

DEPENDS = ("mtah.features.groups.kinematics",)
DEFAULTS = dict(aerial_speed_ms=12.0, aerial_dtheta_deg=5.0, aerial_anorm_ms2=2.0,
                gk_radius_m=1.8, gk_speed_min=1.0, gk_speed_max=4.0, oob_memory_s=10.0,
                box_depth_m=16.5, box_half_width_m=20.16, horizon_s=10.0)


def _since(flag, n, fps, cap_s):
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(flag, idx, -1))
    return np.where(last >= 0, np.minimum((idx - last) / fps, cap_s), cap_s)


@register("phys", context=["phys_aerial", "phys_gk_hold_live"])
def phys(ctx):
    pc = dict(DEFAULTS, **ctx.cfg["features"].get("phys", {}))
    n, fps = ctx.n, ctx.fps
    k = kinematics(ctx)
    speed, dtheta, a_norm = k["kin_speed"], k["kin_dtheta"], k["kin_acc_norm"]
    aerial = ((np.nan_to_num(speed) > pc["aerial_speed_ms"])
              & (np.nan_to_num(dtheta, nan=99.0) < pc["aerial_dtheta_deg"])
              & (np.nan_to_num(a_norm, nan=99.0) < pc["aerial_anorm_ms2"]))
    ground_near = np.where(aerial, np.nan, ctx.team_sign_near)

    # out of play memory
    L, W = ctx.length, ctx.width
    B = ctx.B
    oob = ctx.vis & ((B[:, 0] < 0) | (B[:, 0] > L) | (B[:, 1] < 0) | (B[:, 1] > W))
    t_since_oob = _since(oob, n, fps, 30.0)
    recent_oob = t_since_oob < pc["oob_memory_s"]

    # goalkeeper holding
    phase = filled_period(ctx.data.period)
    hold = {}
    for side, team_idx in (("H", 0), ("A", 1)):
        gk_pos = np.full((n, 2), np.nan)
        gk_sp = np.full(n, np.nan)
        for per, (gk_h, gk_a, _) in ctx.gk_dir.items():
            rows = phase == per
            slot = gk_h if team_idx == 0 else gk_a
            gk_pos[rows] = ctx.P[rows, slot]
            gk_sp[rows] = ctx.sp[rows, slot]
        xc = np.where(ctx.home_pos, gk_pos[:, 0], L - gk_pos[:, 0])       # canonical x
        own_goal_x = 0.0 if team_idx == 0 else L
        in_box = ((np.abs(xc - own_goal_x) < pc["box_depth_m"])
                  & (np.abs(gk_pos[:, 1] - W / 2) < pc["box_half_width_m"]))
        d = np.hypot(gk_pos[:, 0] - B[:, 0], gk_pos[:, 1] - B[:, 1])
        carry = (in_box & (np.nan_to_num(d, nan=99) < pc["gk_radius_m"])
                 & (np.nan_to_num(gk_sp) >= pc["gk_speed_min"]) & (np.nan_to_num(gk_sp) <= pc["gk_speed_max"]))
        hold[side] = carry
    live_h = hold["H"] & ~recent_oob
    live_a = hold["A"] & ~recent_oob
    signed = live_h.astype(np.float32) - live_a.astype(np.float32)
    smooth = lambda x: pd.Series(x.astype(np.float32)).rolling(25, center=True, min_periods=1).mean().to_numpy()  # noqa: E731
    return {
        "phys_aerial": np.where(ctx.vis, aerial.astype(np.float32), np.nan),
        "phys_t_since_aerial": _since(aerial, n, fps, pc["horizon_s"]),
        "phys_near_ground": ground_near,
        "phys_t_since_oob": t_since_oob,
        "phys_gk_hold_live": signed,                          # +1 home keeper, -1 away keeper
        "phys_gk_hold_live_s": smooth(signed),
        "phys_gk_hold_dead_s": smooth((hold["H"] | hold["A"]) & recent_oob),
    }
