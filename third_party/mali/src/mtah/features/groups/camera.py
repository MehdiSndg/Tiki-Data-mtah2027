"""cam (SOURCE-SPECIFIC, Gün 3 / Ek L.14): tactical-camera aim point from the per-frame homography.

The tactical camera follows play, so its aim point is a ball-x proxy that survives the ATD ball
gaps (ball missing on 40.7% of graded frames) and its pan speed is a liveness cue. Only BAR-ATH
has a homography: the group is source-specific (NaN on DFL rows of the joint booster, never in the
net's common schema). Columns, home-attack frame (home attacks +x):
  cam_xc         aim x (m)
  cam_ball_dx    aim x - last visible ball x (m; forward-filled ball)
  cam_ball_adx   |cam_ball_dx|
  cam_pan        |d aim x / dt| over +-0.5 s (m/s)
"""

import numpy as np
import pandas as pd

from ...io.homography import camera_aim_x
from ..registry import register

DEPENDS = ("mtah.io.homography",)


@register("cam", source_specific=True, context=["cam_pan", "cam_xc", "cam_ball_adx"])
def cam(ctx):
    n, L = ctx.n, ctx.length
    if ctx.data.frame[0] != 0 or ctx.data.frame[-1] != n - 1:
        raise ValueError("cam features need the BAR-ATH video-frame axis")
    x = camera_aim_x(n, L)
    xc = np.where(ctx.home_pos, x, L - x)
    ball = pd.Series(np.where(ctx.vis, ctx.xc_b, np.nan)).ffill().to_numpy()
    half = max(1, ctx.fps // 2)
    pan = np.full(n, np.nan)
    pan[half:n - half] = np.abs(x[2 * half:] - x[:n - 2 * half]) / (2 * half / ctx.fps)
    f = {"cam_xc": xc, "cam_ball_dx": xc - ball, "cam_ball_adx": np.abs(xc - ball), "cam_pan": pan}
    return f
