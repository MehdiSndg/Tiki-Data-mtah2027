"""tn (12 Nov): player tempo relative to the period's own level.

Why: the dead-ball evidence leans on player tempo in absolute m/s (pl_p80_speed, pl_mean_speed). A
slow-circulating team in a slow half looks "dead" to a model trained on other providers' matches, and a
fast match hides its stoppages. Here every tempo signal is scaled by robust statistics of the SAME period
of the SAME match (no labels, no ball), so "slow for this half" means the same in every source.

Columns (ball-free; NaN only where no outfield player is tracked)
  tn_p80_rel / tn_mean_rel    p80 / mean outfield speed divided by the period median of that signal
  tn_p80_z                    (p80 - period median) / (period q80 - q20)
  tn_p80_rank                 rank of the p80 speed within the period (0 slowest ... 1 fastest)
  tn_slow_share               share of outfield players slower than half the period's median player speed
"""

import numpy as np

from ..context import nan_call
from ..registry import register


@register("tn", ball_free=True, context=["tn_p80_rel", "tn_p80_rank", "tn_slow_share"])
def tempo_norm(ctx):
    sp = np.where(ctx.outfield, ctx.sp, np.nan)
    p80 = nan_call(np.nanpercentile, sp, 80, axis=1)
    mean = nan_call(np.nanmean, sp, axis=1)
    period = np.asarray(ctx.data.period)
    n = len(p80)
    f = {k: np.full(n, np.nan, np.float32) for k in ("tn_p80_rel", "tn_mean_rel", "tn_p80_z", "tn_p80_rank",
                                                      "tn_slow_share")}
    for per in sorted(set(period.tolist()) - {0}):
        m = period == per
        x, y = p80[m], mean[m]
        ok = ~np.isnan(x)
        if ok.sum() < 50:
            continue
        med, q20, q80 = np.nanmedian(x), np.nanpercentile(x, 20), np.nanpercentile(x, 80)
        f["tn_p80_rel"][m] = x / max(med, 1e-3)
        f["tn_mean_rel"][m] = y / max(np.nanmedian(y), 1e-3)
        f["tn_p80_z"][m] = (x - med) / max(q80 - q20, 1e-3)
        rank = np.full(len(x), np.nan)
        rank[ok] = np.argsort(np.argsort(x[ok], kind="stable"), kind="stable") / max(ok.sum() - 1, 1)
        f["tn_p80_rank"][m] = rank
        ref = 0.5 * np.nanmedian(sp[m])
        with np.errstate(invalid="ignore"):
            slow = np.nansum(sp[m] < ref, axis=1) / np.maximum(np.sum(~np.isnan(sp[m]), axis=1), 1)
        f["tn_slow_share"][m] = np.where(ok, slow, np.nan)
    return f
