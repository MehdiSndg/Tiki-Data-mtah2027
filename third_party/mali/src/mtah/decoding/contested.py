"""Gün 3 (Ek L.10): keep the booster's 25 fps evidence of fast possession exchanges.

The temporal net (5 Hz steps, BiGRU, T-MSE smoothing) erases 8-23 frame opponent spells that the
frame-level booster still sees (P 0.4-0.7). Two local mechanisms, both off by default:

* dynamic_blend: around booster team flips, raise the booster's share of the log-pool
  (w_base -> w_hi within +-radius frames).
* contested mask: frames where the booster gives the team that is NOT the current (reference
  blend) team at least `thr`; the decoder lowers the H<->A turnover penalty at those steps
  (decoder.contested_turnover, see decoding.hsmm).

thr = 0.35 follows the analytic rule p* ~ F1 / 2: a short opponent possession pays off in
expectation when its match probability exceeds about half of the current F1.
"""

import numpy as np

from ..constants import A, D, H


def _log_pool(p_boost, p_net, w):
    """Per-frame weights w (T,) or a scalar; same arithmetic as pipeline.hybrid.log_pool."""
    w = np.broadcast_to(np.asarray(w, np.float64), (len(p_boost),))[:, None]
    lp = w * np.log(np.clip(p_boost, 1e-6, 1)) + (1 - w) * np.log(np.clip(p_net, 1e-6, 1))
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return np.round(p / p.sum(1, keepdims=True), 6)


def dilate(mask, radius):
    """True within +-radius frames of any True frame."""
    if radius <= 0 or not mask.any():
        return mask.copy()
    k = np.ones(2 * radius + 1)
    return np.convolve(mask.astype(np.float64), k, mode="same") > 0


def opponent_prob(p_boost, p_ref):
    """Booster probability of the team that is not p_ref's current team (argmax over H, A)."""
    cur_home = p_ref[:, H] >= p_ref[:, A]
    return np.where(cur_home, p_boost[:, A], p_boost[:, H])


def flip_trigger(p_boost, p_ref, thr=0.35, lookback=10):
    """Frames where the booster's H-vs-A sign changed within the last `lookback` frames and it
    gives the reference blend's opponent at least `thr` (alive frames only)."""
    s = np.sign(p_boost[:, H] - p_boost[:, A])
    n = len(s)
    lo = np.full(n, np.inf)
    hi = np.full(n, -np.inf)
    for k in range(lookback + 1):            # running min / max of the sign over [t - lookback, t]
        sh = np.concatenate([np.full(k, np.nan), s[:n - k]]) if k else s
        lo = np.fmin(lo, sh)
        hi = np.fmax(hi, sh)
    changed = (hi > 0) & (lo < 0)
    alive = p_boost[:, D] < 0.5
    return changed & alive & (opponent_prob(p_boost, p_ref) >= thr)


def dynamic_blend(p_boost, p_net, w_base=0.25, w_hi=0.70, thr=0.35, lookback=10, radius=10):
    """(probs, trigger mask): log-pool with booster weight w_hi within +-radius frames of a
    flip_trigger frame, w_base elsewhere."""
    p_ref = _log_pool(p_boost, p_net, w_base)
    trig = dilate(flip_trigger(p_boost, p_ref, thr, lookback), radius)
    w = np.where(trig, w_hi, w_base)
    return _log_pool(p_boost, p_net, w), trig


def contested_mask(p_boost, p_ref, thr=0.35):
    """Frames where the booster supports the opponent of p_ref's current team with >= thr."""
    return (p_boost[:, D] < 0.5) & (opponent_prob(p_boost, p_ref) >= thr)
