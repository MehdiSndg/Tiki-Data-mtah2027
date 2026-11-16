"""Source-independent preprocessing: velocities, attack direction, goalkeeper slots.

Velocities are recomputed from positions for every source (the ATD speed channel is m/s but the
ball's is mostly empty, and other providers differ), so all features see the same physics.
"""

import warnings

import numpy as np


def velocity(pos, fps, k=2):
    """Central difference over +-k frames; NaN unless both ends are present. pos (T,...,2)."""
    v = np.full(pos.shape, np.nan, np.float32)
    v[k:-k] = (pos[2 * k:] - pos[:-2 * k]) * (fps / (2.0 * k))
    return v


def filled_period(period):
    """Period with zeros (pre-match, half-time, post-match) filled from the nearest period."""
    p = period.astype(np.int16).copy()
    idx = np.where(p > 0, np.arange(len(p)), -1)
    fwd = np.maximum.accumulate(idx)
    rev = np.where(p > 0, np.arange(len(p)), len(p))
    bwd = np.minimum.accumulate(rev[::-1])[::-1]
    fill = np.where(fwd >= 0, p[np.clip(fwd, 0, None)], 0)
    use_bwd = fwd < 0
    fill[use_bwd] = p[np.clip(bwd[use_bwd], 0, len(p) - 1)]
    return fill.astype(np.int8)


def goalkeepers_and_direction(data):
    """{period: (home_gk_slot, away_gk_slot, home_attacks_positive_x)}.

    The goalkeeper of a team in a period is the slot (with >= 20% coverage) whose mean x is
    furthest from the halfway line. Home attacks +x when its goalkeeper stands at the low-x end.
    Works for any source, and recovers slots 24/46 (1-based) on the BAR-ATH ATD.
    """
    length = data.pitch[0]
    out = {}
    x = data.players[:, :, 0]
    for per in sorted(set(int(p) for p in np.unique(data.period)) - {0}):
        rows = data.period == per
        xs = x[rows]
        cover = np.mean(~np.isnan(xs), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_x = np.nanmean(xs, axis=0)
        gks = []
        for t in (0, 1):
            cand = np.flatnonzero((data.team == t) & (cover >= 0.2))
            if len(cand) == 0:
                raise ValueError(f"no well-covered slot for team {t} in period {per}")
            gks.append(int(cand[np.argmax(np.abs(mean_x[cand] - length / 2))]))
        out[per] = (gks[0], gks[1], bool(mean_x[gks[0]] < length / 2))
    return out
