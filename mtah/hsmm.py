"""Two-state explicit-duration decoding of the team call (a hidden semi-Markov model). (P7w)

    from mtah.hsmm import duration_prior, decode_hsmm
    lp = duration_prior(train_labels)                # log P(duration) per frame count, from true possessions
    codes = decode_hsmm(p_home, alive, lp, alpha=1.0, beta=1.0, cost=None)

mtah.teammodel.decode charges a flat (or per-frame) cost per change of team, which is a geometric prior on
how long a possession lasts. Here each possession's length d is scored by beta * lp[d] instead, the
evidence by alpha * log P(team) summed over its frames; cost, if given, is still charged at each start.
Alive stretches are decoded separately (a stoppage ends a possession). A possession longer than DMAX is
continued as the same team at no extra cost (SELF), which keeps very long ones possible.
"""

import numpy as np

FPS = 25
DMAX = 60 * FPS
SELF = 0.0


def duration_prior(label_arrays, dmax=DMAX, smooth=5):
    """log P(d) for d = 0..dmax frames from the true possessions in label_arrays (H/A runs), smoothed."""
    import possession_metrics as pm
    h = np.zeros(dmax + 1)
    for lab in label_arrays:
        for a, b, _ in pm.possession_spells(lab, pm.valid_runs(lab)):
            h[min(b - a, dmax)] += 1
    k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
    h = np.convolve(h, k, mode="same") + 1e-3
    h[0] = 1e-12
    return np.log(h / h.sum())


def _run(e, lp, cost, dmax):
    """e: (n, 2) per-frame log evidence; returns the best state per frame."""
    n = len(e)
    C = np.vstack([np.zeros((1, 2)), np.cumsum(e, axis=0)])
    best = np.full((n + 1, 2), -np.inf)
    best[0] = 0.0
    arg_d = np.zeros((n + 1, 2), np.int64)
    arg_p = np.zeros((n + 1, 2), np.int64)
    for t in range(1, n + 1):
        d = np.arange(1, min(dmax, t) + 1)
        s = t - d
        for k in (0, 1):
            seg = C[t, k] - C[s, k] + lp[d]
            other = best[s, 1 - k] - np.where(s > 0, cost[np.minimum(s, n - 1)], 0.0)
            same = np.where(d == dmax, best[s, k] - SELF, -np.inf)
            start = np.where(s == 0, 0.0, np.maximum(other, same))
            v = seg + start
            i = int(np.argmax(v))
            best[t, k], arg_d[t, k] = v[i], d[i]
            arg_p[t, k] = 1 - k if (s[i] == 0 or other[i] >= same[i]) else k
    out = np.zeros(n, np.int8)
    t, k = n, int(np.argmax(best[n]))
    while t > 0:
        d = arg_d[t, k]
        out[t - d:t] = k
        t, k = t - d, arg_p[t, k]
    return out


def decode_hsmm(p_home, alive, lp, alpha=1.0, beta=1.0, cost=None, dmax=DMAX):
    import possession_metrics as pm
    p = np.clip(np.asarray(p_home, np.float64), 1e-4, 1 - 1e-4)
    e = alpha * np.stack([np.log(p), np.log(1 - p)], axis=1)
    c = np.zeros(len(p)) if cost is None else np.broadcast_to(np.asarray(cost, np.float64), p.shape)
    out = np.full(len(p), pm.D, np.int8)
    edges = np.diff(np.r_[0, alive.astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        st = _run(e[a:z], beta * lp, c[a:z], dmax)
        out[a:z] = np.where(st == 0, pm.H, pm.A)
    return out
