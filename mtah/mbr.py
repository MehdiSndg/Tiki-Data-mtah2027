"""Vendored unchanged from Mehmet Ali Topkara's delivery (team/mehmet_ali_topkara, code/src/mtah/decoding/mbr.py).

Minimum-Bayes-risk decoding for the spell metric (11 Nov, `decoder.mbr`).

The Viterbi decoder returns the single most probable segmentation. The ranking metric is not the
probability of the whole path: it is the F1 of possession spells matched by temporal IoU. So this
decoder
  1. samples `samples` segmentations from the same semi-Markov model (scores / `temperature`;
     forward sums instead of maxima, backward sampling),
  2. gives every H / A spell seen in a sample (plus the Viterbi path's) a credit: the mean over the
     samples and over the metric's IoU thresholds of "a same-team spell with IoU >= threshold exists",
  3. picks the non-overlapping set of spells with the highest expected F1
     2 * sum(credit) / (n_chosen + mean spells per sample)   (Dinkelbach iterations + interval DP).
Steps between chosen spells are dead ball. Everything works on decoder steps; the caller applies the
usual frame-level boundary refinement. Deterministic for a fixed `seed`.

Only the model's own probabilities are used: no labels of any match.
"""

import numpy as np

try:
    from numba import njit
except ImportError:                      # the sampler is too slow without numba; keep importable
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

NEG = -1e300
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)         # possession_metrics.SPELL_IOUS


@njit(cache=True)
def _lse(x, n):
    m = NEG
    for i in range(n):
        if x[i] > m:
            m = x[i]
    if m <= NEG / 2:
        return NEG
    s = 0.0
    for i in range(n):
        if x[i] > NEG / 2:
            s += np.exp(x[i] - m)
    return m + np.log(s)


@njit(cache=True)
def _seg_score(start, C, s, t, k, S, d, dur_lp, dur_edge, dmin, cstep, dur_c, inv_t):
    edge = (s == 0) or (t == S)
    if d < dmin[k] and not edge:
        return NEG
    prev = start[s, k]
    if prev <= NEG / 2:
        return NEG
    dur = dur_edge[k] if edge else dur_lp[k, d]
    if (not edge) and k != 2 and cstep[s] > 0 and dur_c[d] > dur:
        dur = dur_c[d]
    if dur <= NEG / 2:
        return NEG
    return prev + (C[t, k] - C[s, k] + dur) * inv_t


@njit(cache=True)
def _trans_score(trans, adj, t, kp, k, inv_t):
    v = trans[kp, k]
    if kp + k == 1:
        v += adj[t, 0]
    elif k == 2:
        v += adj[t, 1]
    else:
        v += adj[t, 2]
    return v * inv_t


@njit(cache=True)
def _pick(w, n, u):
    """Index sampled from log-weights w[:n] with the uniform draw u."""
    tot = _lse(w, n)
    acc = 0.0
    last = -1
    for i in range(n):
        if w[i] > NEG / 2:
            acc += np.exp(w[i] - tot)
            last = i
            if u < acc:
                return i
    return last


@njit(cache=True)
def sample_paths(step_lp, dur_lp, dur_edge, dmin, trans, bonus, adj, cstep, dur_c, temperature, n, seed):
    """(n, S) int8 step labels sampled from exp(score / temperature) of the decoder's model."""
    S, K = step_lp.shape
    dmax = dur_lp.shape[1] - 1
    inv_t = 1.0 / temperature
    C = np.zeros((S + 1, K))
    for t in range(S):
        for k in range(K):
            C[t + 1, k] = C[t, k] + step_lp[t, k]
    end = np.full((S + 1, K), NEG)       # log-sum of all segmentations of [0, t) whose last segment is k
    start = np.full((S + 1, K), NEG)     # ... and a segment of state k starts at t
    for k in range(K):
        start[0, k] = 0.0
    buf = np.empty(max(dmax, K) + 1)
    for t in range(1, S + 1):
        dm = dmax if dmax < t else t
        for k in range(K):
            for d in range(1, dm + 1):
                buf[d - 1] = _seg_score(start, C, t - d, t, k, S, d, dur_lp, dur_edge, dmin, cstep, dur_c, inv_t)
            end[t, k] = _lse(buf, dm)
        if t < S:
            for k in range(K):
                for kp in range(K):
                    buf[kp] = NEG if (kp == k or end[t, kp] <= NEG / 2) else \
                        end[t, kp] + _trans_score(trans, adj, t, kp, k, inv_t)
                v = _lse(buf, K)
                start[t, k] = v + bonus[t] * inv_t if v > NEG / 2 else NEG
    np.random.seed(seed)
    out = np.zeros((n, S), np.int8)
    for i in range(n):
        for k in range(K):
            buf[k] = end[S, k]
        k = _pick(buf, K, np.random.random())
        t = S
        while t > 0:
            dm = dmax if dmax < t else t
            for d in range(1, dm + 1):
                buf[d - 1] = _seg_score(start, C, t - d, t, k, S, d, dur_lp, dur_edge, dmin, cstep, dur_c, inv_t)
            d = _pick(buf, dm, np.random.random()) + 1
            for j in range(t - d, t):
                out[i, j] = k
            t -= d
            if t == 0:
                break
            for kp in range(K):
                buf[kp] = NEG if (kp == k or end[t, kp] <= NEG / 2) else \
                    end[t, kp] + _trans_score(trans, adj, t, kp, k, inv_t)
            k = _pick(buf, K, np.random.random())
    return out


def spells(labels):
    """(m, 3) int64 [start, stop, team] of the maximal H / A runs of a step-label vector."""
    lab = np.asarray(labels)
    cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    a = np.r_[0, cut]
    b = np.r_[cut, len(lab)]
    k = lab[a]
    keep = k < 2
    return np.stack([a[keep], b[keep], k[keep]], axis=1).astype(np.int64)


@njit(cache=True)
def _credits(cand, starts, stops, teams, offs, ious):
    """Mean over samples and IoU thresholds of 'a same-team spell with IoU >= threshold exists'."""
    n_s = len(offs) - 1
    out = np.zeros(len(cand))
    for c in range(len(cand)):
        a, b, k = cand[c, 0], cand[c, 1], cand[c, 2]
        tot = 0.0
        for s in range(n_s):
            lo, hi = offs[s], offs[s + 1]
            l, h = lo, hi                          # first spell with stop > a (stops are increasing)
            while l < h:
                m = (l + h) // 2
                if stops[m] > a:
                    h = m
                else:
                    l = m + 1
            best = 0.0
            j = l
            while j < hi and starts[j] < b:
                if teams[j] == k:
                    inter = min(b, stops[j]) - max(a, starts[j])
                    iou = inter / ((b - a) + (stops[j] - starts[j]) - inter)
                    if iou > best:
                        best = iou
                j += 1
            for q in range(len(ious)):
                if best >= ious[q]:
                    tot += 1.0
        out[c] = tot / (n_s * len(ious))
    return out


@njit(cache=True)
def _select(cand, value, S):
    """Indices of the non-overlapping candidates with the largest total value; two chosen spells of
    the same team never touch (they would merge into one spell). cand sorted by (stop, start)."""
    n = len(cand)
    E = np.full((S + 1, 2), NEG)          # best total with a chosen spell of team k ending exactly at t
    Ei = np.full((S + 1, 2), -1, np.int64)
    F = np.zeros(S + 1)                   # best total using steps [0, t)
    Fi = np.full(S + 1, -1, np.int64)
    prev = np.full(n, -1, np.int64)
    c = 0
    for t in range(1, S + 1):
        while c < n and cand[c, 1] == t:
            a, k = cand[c, 0], cand[c, 2]
            base, bp = 0.0, -1
            if a >= 1 and F[a - 1] > base:
                base, bp = F[a - 1], Fi[a - 1]
            if E[a, 1 - k] > base:
                base, bp = E[a, 1 - k], Ei[a, 1 - k]
            v = base + value[c]
            if v > E[t, k]:
                E[t, k] = v
                Ei[t, k] = c
                prev[c] = bp
            c += 1
        F[t], Fi[t] = F[t - 1], Fi[t - 1]
        for k in range(2):
            if E[t, k] > F[t]:
                F[t], Fi[t] = E[t, k], Ei[t, k]
    chosen = []
    i = Fi[S]
    while i >= 0:
        chosen.append(i)
        i = prev[i]
    return np.array(chosen[::-1], np.int64)


def choose(samples, viterbi_labels=None, max_iter=20):
    """(step labels with the expected-F1-optimal spells and D elsewhere, info dict)."""
    S = samples.shape[1]
    per = [spells(s) for s in samples]
    offs = np.r_[0, np.cumsum([len(p) for p in per])].astype(np.int64)
    allsp = np.concatenate(per) if offs[-1] else np.zeros((0, 3), np.int64)
    pool = [allsp] + ([spells(viterbi_labels)] if viterbi_labels is not None else [])
    cand = np.unique(np.concatenate(pool), axis=0)
    out = np.full(S, 2, np.int8)
    if len(cand) == 0:
        return out, dict(n_candidates=0, n_chosen=0, expected_f1=0.0, mean_spells=0.0)
    cand = cand[np.lexsort((cand[:, 0], cand[:, 1]))]
    credit = _credits(cand, allsp[:, 0].copy(), allsp[:, 1].copy(), allsp[:, 2].copy(), offs,
                      np.array(IOUS))
    n_ref = float(np.mean([len(p) for p in per]))
    f, chosen = 0.5, np.zeros(0, np.int64)
    for _ in range(max_iter):
        chosen = _select(cand, credit - f / 2.0, S)
        f_new = 2.0 * float(credit[chosen].sum()) / (len(chosen) + n_ref) if len(chosen) + n_ref > 0 else 0.0
        if abs(f_new - f) < 1e-9:
            break
        f = f_new
    for a, b, k in cand[chosen]:
        out[a:b] = k
    return out, dict(n_candidates=int(len(cand)), n_chosen=int(len(chosen)), expected_f1=round(f, 6),
                     mean_spells=round(n_ref, 3))


def decode(step_lp, dur_lp, dur_edge, dmin, trans, bonus, adj, cstep, dur_c, spec, viterbi_labels=None):
    """Step labels of the MBR decision. spec: decoder.mbr = {samples, temperature, seed}."""
    samples = sample_paths(np.ascontiguousarray(step_lp, np.float64), dur_lp, dur_edge, dmin, trans, bonus, adj,
                           cstep, dur_c, float(spec.get("temperature", 1.0)), int(spec.get("samples", 300)),
                           int(spec.get("seed", 2027)))
    return choose(samples, viterbi_labels)[0].astype(np.int64)
