"""Cross-ATD ball fusion: the broadcast recording's ATD ball mapped onto the tactical timeline.

Alignment (Gün 2 madenciliği, MASTER_PLAN Ek G): slots are tracklets, not people, and the
broadcast camera shows only part of the pitch, so neither slot ids nor team centroids align the
two recordings. The working signal is a permutation-invariant one-sided chamfer distance: every
player visible on broadcast to the nearest tactical player. Per 60 s window the lag minimising
its median is found (coarse -> 1 frame). Result on BAR-ATH: 84/95 windows align with median
error 0.28 m; the lag is constant per half (4013-4015 frames in H1, 26919-26920 in H2); the
rest are cuts/replays and are not used.

fused_ball(tac) -> (T, 2) metres, NaN where broadcast has no aligned ball. Filters: inside the
pitch +- 3 m, and no 40+ m/s jumps to a neighbouring fused frame. On graded frames where the
tactical ball is missing this recovers ~10% (6.7k frames, half of them in > 10 s gaps); where
both see the ball they agree within 2 m on 73% of frames (median 0.8 m).
"""

import numpy as np

from .. import constants as C
from ..paths import CACHE, OFFICIAL
from .atd import read_metadata, read_raw

BROAD = OFFICIAL / "data" / "broadcast"
WIN = 1_500


def lag_error(pb, tp, idx, lag):
    errs = []
    for i in idx:
        j = i + lag
        if j < 0 or j >= len(tp):
            continue
        b = pb[i][np.isfinite(pb[i, :, 0])]
        t = tp[j][np.isfinite(tp[j, :, 0])]
        if len(b) < 5 or len(t) < 10:
            continue
        errs.append(np.median(np.linalg.norm(b[:, None] - t[None], axis=-1).min(1)))
    return float(np.median(errs)) if len(errs) >= 8 else np.inf


def best_lag(pb, tp, idx, lags):
    err, lag = min((lag_error(pb, tp, idx, lag), lag) for lag in lags)
    return lag, err


def load_broadcast():
    slot_team, _, _ = read_metadata(BROAD / "BAR-ATH_broadcast_atd.xml")
    _, players, ball = read_raw(BROAD / "BAR-ATH_broadcast_atd.txt", len(slot_team))
    scale = np.array(C.PITCH, np.float32)
    return players[:, :, :2] * scale, ball[:, :2] * scale


def align(pb, tp, max_err=1.0):
    """(lag per broadcast frame (NaN = unaligned), [(window start, lag, err)])."""
    rows, lag_of, prev = [], np.full(len(pb), np.nan), None
    for a in range(0, len(pb) - WIN, WIN):
        idx = np.arange(a, a + WIN, 75)
        if np.isfinite(pb[idx, :, 0]).sum(1).mean() < 5:
            continue
        coarse = range(-20_000, 50_000, 100) if prev is None else range(prev - 1_500, prev + 1_501, 50)
        lag, err = best_lag(pb, tp, idx, coarse)
        if err > 2.5 and prev is not None:
            lag, err = best_lag(pb, tp, idx, range(-20_000, 50_000, 100))
        lag, err = best_lag(pb, tp, np.arange(a, a + WIN, 25), range(lag - 60, lag + 61, 4))
        lag, err = best_lag(pb, tp, np.arange(a, a + WIN, 10), range(lag - 4, lag + 5))
        rows.append((a, int(lag), err))
        if err < max_err:
            lag_of[a:a + WIN] = lag
            prev = lag
    return lag_of, rows


def _fuse(tac, pb, bb, lag_of):
    fused = np.full((tac.n_frames, 2), np.nan, np.float32)
    src = np.flatnonzero(np.isfinite(lag_of) & np.isfinite(bb[:, 0]))
    dst = src + lag_of[src].astype(int)
    ok = (dst >= 0) & (dst < tac.n_frames)
    fused[dst[ok]] = bb[src[ok]]
    L, W = tac.pitch
    out = (fused[:, 0] < -3) | (fused[:, 0] > L + 3) | (fused[:, 1] < -3) | (fused[:, 1] > W + 3)
    fused[out] = np.nan
    step = np.linalg.norm(np.diff(fused, axis=0), axis=1) * tac.fps
    jump = np.nan_to_num(step, nan=0.0) > 40.0
    bad = np.r_[jump, False] | np.r_[False, jump]
    fused[bad] = np.nan
    return fused


def fused_ball(tac, rebuild=False):
    path = CACHE / "broadcast_ball_on_tactical.npy"
    if path.is_file() and not rebuild:
        return np.load(path)
    pb, bb = load_broadcast()
    lag_of, _ = align(pb, tac.players)
    fused = _fuse(tac, pb, bb, lag_of)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, fused)
    return fused


def fill_ball(tac):
    """Copy of `tac` named '<name>_bf' with missing ball frames filled from the broadcast."""
    out = tac.copy()
    fused = fused_ball(tac)
    miss = np.isnan(out.ball[:, 0]) & np.isfinite(fused[:, 0])
    out.ball[miss] = fused[miss]
    out.name = f"{tac.name}_bf"
    out.meta = dict(tac.meta, broadcast_filled_frames=int(miss.sum()))
    return out


def bar_data(cfg):
    """BAR-ATH tracking as the config wants it (data.broadcast_fill -> broadcast-filled ball).
    The dataset name doubles as the feature-cache key."""
    from .atd import cached_barath
    tac = cached_barath()
    return fill_ball(tac) if cfg.get("data", {}).get("broadcast_fill", False) else tac
