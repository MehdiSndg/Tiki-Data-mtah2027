"""geo (Group 2): collective team geometry. Ball-free, so it works through every ball gap.

Per team (outfield players only, attack coordinates: own goal at 0, opponent goal at pitch length):
width / length (p90 - p10), compactness (mean distance to the team centroid), last line and front
line heights, share of players in the attacking / defensive third, collective lateral velocity and
movement synchrony (|mean velocity| / mean |velocity|). Columns are home-minus-away where a
difference is meaningful, so the sign reads "home-ish".
"""

import numpy as np

from ..context import nan_call
from ..registry import register


def _team_stats(ax, y, vx, vy, mask, length):
    """ax: attack coordinate, y, velocities (T,N); mask (T,N) of this team's outfield players."""
    ax = np.where(mask, ax, np.nan)
    yy = np.where(mask, y, np.nan)
    q_ax = nan_call(np.nanpercentile, ax, [10, 90], axis=1)
    q_y = nan_call(np.nanpercentile, yy, [10, 90], axis=1)
    cx = nan_call(np.nanmean, ax, axis=1)
    cy = nan_call(np.nanmean, yy, axis=1)
    compact = nan_call(np.nanmean, np.hypot(ax - cx[:, None], yy - cy[:, None]), axis=1)
    count = np.sum(mask & ~np.isnan(ax), axis=1).astype(np.float32)
    att3 = np.sum(ax > 2 * length / 3, axis=1) / np.maximum(1, count)
    def3 = np.sum(ax < length / 3, axis=1) / np.maximum(1, count)
    mvx = nan_call(np.nanmean, np.where(mask, vx, np.nan), axis=1)
    mvy = nan_call(np.nanmean, np.where(mask, vy, np.nan), axis=1)
    mspeed = nan_call(np.nanmean, np.where(mask, np.hypot(vx, vy), np.nan), axis=1)
    sync = np.hypot(mvx, mvy) / np.maximum(mspeed, 0.1)
    return dict(width=q_y[1] - q_y[0], length=q_ax[1] - q_ax[0], compact=compact,
                last=nan_call(np.nanmin, ax, axis=1), front=q_ax[1], att3=att3, def3=def3,
                vy=mvy, sync=sync, cx=cx)


@register("geo", ball_free=True,
          context=["geo_width_diff", "geo_length_diff", "geo_compact_diff", "geo_cen_sep",
                   "geo_sync_diff"])
def geometry(ctx):
    L = ctx.length
    vy = ctx.vp[..., 1]
    h = _team_stats(ctx.xc_p, ctx.y_p, ctx.vx_c, vy, ctx.outfield_home, L)
    a = _team_stats(L - ctx.xc_p, ctx.y_p, -ctx.vx_c, vy, ctx.outfield_away, L)
    out = {}
    for key in ("width", "length", "compact", "last", "front", "att3", "def3", "sync"):
        out[f"geo_{key}_H"] = h[key]
        out[f"geo_{key}_A"] = a[key]
        out[f"geo_{key}_diff"] = h[key] - a[key]
    out["geo_area_logratio"] = np.log((h["width"] * h["length"] + 1.0) / (a["width"] * a["length"] + 1.0))
    # Centroids in the common canonical frame: home centroid minus away centroid along +x.
    out["geo_cen_sep"] = h["cx"] - (L - a["cx"])
    out["geo_block_xc"] = (h["cx"] + (L - a["cx"])) / 2
    out["geo_vy_diff"] = h["vy"] - a["vy"]
    return out
