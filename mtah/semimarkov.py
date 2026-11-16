"""Explicit-duration semi-Markov decoder, vendored unchanged from Mehmet Ali Topkara's delivery (team branch
team/mehmet_ali_topkara, commit aa7f1fcd, code/src/mtah/decoding/hsmm.py) except for the two lines marked VENDORED: the
label constants are defined here instead of imported from his package, and the optional expected-F1 step (decoder.mbr,
not used) is not available. CONFIG is his externally tuned decoder (main_ext_met_dt) with the four values P9dh picked
on DFL + Metrica for our probabilities (emission scale 4, boundary gamma 6, turnover penalty 10, duration weight 2).

Original module docstring:

3-state (H, A, D) explicit-duration semi-Markov Viterbi decoder.

Segment score = sum of frame log-emissions + alpha * log P_dur(length | state)
                + log transition(prev -> state) [+ boundary bonus at the segment start].
Self-transitions are forbidden (a state's persistence is its duration), so H D H is two
possessions exactly as the metric counts them. Durations have a hard minimum (except for
segments truncated by the start/end of a run) and a plateau so very long possessions are not
penalised further. Decoding runs on blocks of `stride` frames, then every boundary is
re-optimised at frame resolution within +-refine_frames.
"""

import math

import numpy as np

H, A, D = 0, 1, 2          # VENDORED: possession_metrics' codes (his constants.py has the same)

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:          # pure-numpy fallback below gives identical results, ~10x slower
    HAVE_NUMBA = False

NEG = -1e300


def _viterbi_loops(step_lp, dur_lp, dur_edge, dmin, trans, bonus, adj, cstep, dur_c):
    S, K = step_lp.shape
    dmax_all = dur_lp.shape[1] - 1
    C = np.zeros((S + 1, K))
    for t in range(S):
        for k in range(K):
            C[t + 1, k] = C[t, k] + step_lp[t, k]
    dp = np.full((S + 1, K), NEG)
    back_d = np.zeros((S + 1, K), np.int64)
    back_k = np.full((S + 1, K), -1, np.int64)
    best_prev = np.full((S + 1, K), NEG)
    best_prev_k = np.full((S + 1, K), -1, np.int64)
    for k in range(K):
        best_prev[0, k] = 0.0
    for t in range(1, S + 1):
        for k in range(K):
            best = NEG
            bd = 0
            bk = -1
            dm = dmax_all if dmax_all < t else t
            for d in range(1, dm + 1):
                s = t - d
                edge = (s == 0) or (t == S)
                if d < dmin[k] and not edge:
                    continue
                prev = best_prev[s, k]
                if prev <= NEG / 2:
                    continue
                dur = dur_edge[k] if edge else dur_lp[k, d]
                if (not edge) and k != 2 and cstep[s] > 0:   # contested start: duration mixture
                    if dur_c[d] > dur:
                        dur = dur_c[d]
                if dur <= NEG / 2:
                    continue
                v = prev + C[t, k] - C[s, k] + dur
                if v > best:
                    best = v
                    bd = d
                    bk = best_prev_k[s, k] if s > 0 else -1
            dp[t, k] = best
            back_d[t, k] = bd
            back_k[t, k] = bk
        if t < S:
            for k in range(K):
                bp = NEG
                bpk = -1
                for kp in range(K):
                    if kp == k or dp[t, kp] <= NEG / 2:
                        continue
                    v = dp[t, kp] + trans[kp, k]
                    if kp + k == 1:            # local adjustments: H <-> A turnover
                        v += adj[t, 0]
                    elif k == 2:               # H / A -> D stoppage
                        v += adj[t, 1]
                    else:                      # D -> H / A restart
                        v += adj[t, 2]
                    if v > bp:
                        bp = v
                        bpk = kp
                best_prev[t, k] = bp + bonus[t] if bpk >= 0 else NEG
                best_prev_k[t, k] = bpk
    labels = np.zeros(S, np.int64)
    k = 0
    for kk in range(1, K):
        if dp[S, kk] > dp[S, k]:
            k = kk
    t = S
    while t > 0:
        d = back_d[t, k]
        pk = back_k[t, k]
        for i in range(t - d, t):
            labels[i] = k
        t -= d
        k = pk
        if k < 0:
            break
    return labels


def _viterbi_numpy(step_lp, dur_lp, dur_edge, dmin, trans, bonus, adj, cstep, dur_c):
    """Same recursion as _viterbi_loops, vectorised over durations."""
    S, K = step_lp.shape
    dmax_all = dur_lp.shape[1] - 1
    C = np.vstack([np.zeros((1, K)), np.cumsum(step_lp, axis=0)])
    dp = np.full((S + 1, K), NEG)
    back_d = np.zeros((S + 1, K), np.int64)
    back_k = np.full((S + 1, K), -1, np.int64)
    best_prev = np.full((S + 1, K), NEG)
    best_prev_k = np.full((S + 1, K), -1, np.int64)
    best_prev[0] = 0.0
    for t in range(1, S + 1):
        d = np.arange(1, min(dmax_all, t) + 1)
        s = t - d
        edge = (s == 0) | (t == S)
        for k in range(K):
            prev = best_prev[s, k]
            dur = np.where(edge, dur_edge[k], dur_lp[k, d])
            if k != D:
                dur = np.where((cstep[s] > 0) & ~edge, np.maximum(dur, dur_c[d]), dur)
            v = prev + (C[t, k] - C[s, k]) + dur
            bad = ((d < dmin[k]) & ~edge) | (prev <= NEG / 2) | (dur <= NEG / 2)
            v[bad] = NEG
            i = int(np.argmax(v))          # first maximum == shortest duration, as in the loop
            if v[i] > NEG:
                dp[t, k] = v[i]
                back_d[t, k] = d[i]
                back_k[t, k] = best_prev_k[s[i], k] if s[i] > 0 else -1
        if t < S:
            for k in range(K):
                cand = dp[t].copy() + trans[:, k]
                if k != D:
                    cand[1 - k] += adj[t, 0]               # turnover
                    cand[D] += adj[t, 2]                   # restart
                else:
                    cand[H] += adj[t, 1]                   # stoppage
                    cand[A] += adj[t, 1]
                cand[k] = NEG
                cand[dp[t] <= NEG / 2] = NEG
                kp = int(np.argmax(cand))
                if cand[kp] > NEG / 2:
                    best_prev[t, k] = cand[kp] + bonus[t]
                    best_prev_k[t, k] = kp
    labels = np.zeros(S, np.int64)
    k = int(np.argmax(dp[S]))
    t = S
    while t > 0:
        dd = back_d[t, k]
        pk = back_k[t, k]
        labels[t - dd:t] = k
        t -= dd
        k = pk
        if k < 0:
            break
    return labels


_viterbi = njit(cache=True)(_viterbi_loops) if HAVE_NUMBA else _viterbi_numpy


def duration_logprior(spec, stride, alpha, dmax_steps):
    """alpha * log-normal density over segment lengths (in frames), as a table over steps."""
    out = np.full(dmax_steps + 1, NEG)
    sigma, median = float(spec["sigma"]), float(spec["median"])
    for d in range(1, dmax_steps + 1):
        frames = d * stride
        if frames < spec["min"] or frames > spec["max"]:
            continue
        x = min(frames, spec["plateau"])
        lp = -math.log(x * sigma * math.sqrt(2 * math.pi)) - (math.log(x) - math.log(median)) ** 2 / (2 * sigma ** 2)
        out[d] = alpha * lp
    return out


def _tables(cfg):
    dc = cfg["decoder"]
    stride = int(dc["stride"])
    poss, dead = dc["duration"]["possession"], dc["duration"]["dead"]
    dmax = int(math.ceil(max(poss["max"], dead["max"]) / stride))
    lp_poss = duration_logprior(poss, stride, dc["alpha_duration"], dmax)
    lp_dead = duration_logprior(dead, stride, dc["alpha_duration"], dmax)
    dur = np.stack([lp_poss, lp_poss, lp_dead])
    edge = np.array([lp_poss[lp_poss > NEG / 2].max(), lp_poss[lp_poss > NEG / 2].max(),
                     lp_dead[lp_dead > NEG / 2].max()])
    dmin = np.array([math.ceil(poss["min"] / stride), math.ceil(poss["min"] / stride),
                     math.ceil(dead["min"] / stride)], np.int64)
    trans = np.full((3, 3), NEG)
    trans[H, A] = trans[A, H] = -dc["penalty_turnover"]
    trans[H, D] = trans[A, D] = -dc["penalty_stoppage"]
    trans[D, H] = trans[D, A] = -dc["penalty_restart"]
    return stride, dur, edge, dmin, trans


def emission_logprob(probs, cfg):
    """Frame log-emissions. decoder.emission_scale (beta, default 1.0) scales them against the
    duration / transition priors: beta > 1 lets the probabilities override the priors more often
    (sharper; the 5-seed average acts like a temperature T > 1)."""
    dc = cfg["decoder"]
    p = np.clip(np.asarray(probs, np.float64), dc["prob_floor"], 1.0)
    lp = np.log(p) * np.asarray(dc["emission_weights"], np.float64)
    beta = float(dc.get("emission_scale", 1.0))
    if beta != 1.0:
        lp = lp * beta
    return np.round(lp, 6)            # determinism across CPUs (MASTER_PLAN C7)


def boundary_bonus(boundary, stride, n_steps, gamma):
    """Per-step segment-start bonus gamma * logit(b) from the network's boundary head.
    boundary: per-frame P(a segment starts in this step) repeated over the step's frames (the
    run's first frame = step 0), so step t reads frame t * stride. Returns (n_steps + 1,)."""
    bonus = np.zeros(n_steps + 1)
    if boundary is None or gamma == 0.0:
        return bonus
    b = np.clip(np.asarray(boundary, np.float64)[::stride][:n_steps], 1e-4, 1 - 1e-4)
    bonus[:len(b)] = np.round(gamma * np.log(b / (1.0 - b)), 6)
    bonus[0] = 0.0                    # the first segment of a run starts there anyway
    return bonus


def _step_mean(x, stride, n_steps):
    out = np.zeros(n_steps)
    if x is None:
        return out
    c = np.zeros(n_steps * stride)
    m = min(len(x), len(c))
    c[:m] = np.nan_to_num(np.asarray(x, np.float64)[:m])
    if m < len(c):
        c[m:] = c[m - 1] if m else 0.0
    return np.round(c.reshape(n_steps, stride).mean(1), 6)


def _contested_steps(contested, stride, n_steps):
    c = np.zeros(n_steps * stride, bool)
    m = min(len(contested), len(c))
    c[:m] = np.asarray(contested, bool)[:m]
    step = c.reshape(n_steps, stride).any(1)
    return step | np.r_[step[1:], False] | np.r_[False, step[:-1]]


def contested_duration(contested, stride, n_steps, cfg, dmax_steps):
    """(cstep (S+1,) int64, dur_c table): H / A segments that START at a contested step may use
    the short 'contested possession' duration prior decoder.contested_duration (log-normal spec
    like decoder.duration.possession) when it scores higher than the controlled-possession prior:
    a two-component duration mixture (Viterbi max). Off when the key is unset."""
    dc = cfg["decoder"]
    cstep = np.zeros(n_steps + 1, np.int64)
    dur_c = np.full(dmax_steps + 1, NEG)
    spec = dc.get("contested_duration")
    if contested is None or not spec:
        return cstep, dur_c
    full = dict(dc["duration"]["possession"], **spec)
    dur_c = duration_logprior(full, stride, dc["alpha_duration"], dmax_steps)
    cstep[:n_steps] = _contested_steps(contested, stride, n_steps)
    return cstep, dur_c


def turnover_adjust(contested, stride, n_steps, cfg):
    """Per-step bonus on H <-> A transitions (Gün 3, Ek L.10): at steps touching a contested
    frame (+-1 step) the turnover penalty becomes decoder.contested_turnover instead of
    decoder.penalty_turnover. Zeros when contested is None or the key is unset."""
    out = np.zeros(n_steps + 1)
    ct = cfg["decoder"].get("contested_turnover")
    if contested is None or ct is None:
        return out
    step = _contested_steps(contested, stride, n_steps)
    out[:n_steps][step] = float(cfg["decoder"]["penalty_turnover"]) - float(ct)
    return out


def _refine(labels, lp, radius):
    """Move every boundary to the frame maximising the two adjacent segments' emission sum."""
    out = labels.copy()
    cum = np.vstack([np.zeros((1, lp.shape[1])), np.cumsum(lp, axis=0)])
    change = np.flatnonzero(out[1:] != out[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(out)]))
    for i in range(1, len(bounds) - 1):
        a, b, c = bounds[i - 1], bounds[i], bounds[i + 1]
        k1, k2 = out[a], out[b]
        lo, hi = max(a + 1, b - radius), min(c - 1, b + radius)
        if hi <= lo:
            continue
        cand = np.arange(lo, hi + 1)
        score = cum[cand, k1] - cum[cand, k2]
        nb = int(cand[np.argmax(score)])
        out[min(b, nb):max(b, nb)] = k1 if nb > b else k2
        bounds[i] = nb
    return out


def decode_run(lp, cfg, forced_dead=None, boundary=None, contested=None, regime=None):
    """Decode one run. lp: (T,3) frame log-emissions. forced_dead: bool (T,) or None.
    boundary: per-frame boundary probability (T,) or None; used only when
    decoder.boundary_gamma != 0 (see boundary_bonus). contested: per-frame bool (T,) or None;
    used only when decoder.contested_turnover is set (see turnover_adjust). regime (Gün 4, Ek M.5):
    optional dict of per-frame arrays -- 'stoppage_adj' / 'restart_adj' (log-score added to
    H/A -> D and D -> H/A transitions, i.e. minus the penalty change; input-driven HSMM) and
    'refine_lp' ((T, 3) log-scores the +-refine_frames boundary snap uses instead of lp)."""
    lp = lp.copy()
    if forced_dead is not None and forced_dead.any():
        lp[forced_dead, H] = -1e4
        lp[forced_dead, A] = -1e4
        lp[forced_dead, D] = 0.0
    stride, dur, edge, dmin, trans = _tables(cfg)
    T = len(lp)
    S = int(math.ceil(T / stride))
    pad = S * stride - T
    lp_pad = np.vstack([lp, np.repeat(lp[-1:], pad, axis=0)]) if pad else lp
    step_lp = lp_pad.reshape(S, stride, 3).sum(axis=1)
    bonus = boundary_bonus(boundary, stride, S, float(cfg["decoder"].get("boundary_gamma", 0.0)))
    adj = np.zeros((S + 1, 3))
    adj[:, 0] = turnover_adjust(contested, stride, S, cfg)
    if regime is not None:
        adj[:S, 1] = _step_mean(regime.get("stoppage_adj"), stride, S)
        adj[:S, 2] = _step_mean(regime.get("restart_adj"), stride, S)
        if regime.get("turnover_adj") is not None:            # Gün 5 contact gate (decoding.contact)
            adj[:S, 0] += _step_mean(regime.get("turnover_adj"), stride, S)
    cstep, dur_c = contested_duration(contested, stride, S, cfg, dur.shape[1] - 1)
    step_labels = _viterbi(step_lp, dur, edge, dmin, trans, bonus, adj, cstep, dur_c)
    spec = cfg["decoder"].get("mbr")
    if spec:                                  # 11 Nov: expected-spellF1 decision over sampled paths (decoding.mbr)
        raise NotImplementedError('decoder.mbr is not vendored')          # VENDORED
    labels = np.repeat(step_labels, stride)[:T].astype(np.int8)
    ref = lp if regime is None or regime.get("refine_lp") is None else regime["refine_lp"]
    labels = _refine(labels, ref, int(cfg["decoder"]["refine_frames"]))
    if forced_dead is not None:
        labels[forced_dead] = D
    return labels


CONFIG = {"decoder": {"stride": 5, "prob_floor": 0.0001, "emission_weights": [1.0, 1.0, 1.15], "alpha_duration": 2.0,
                      "duration": {"possession": {"median": 150, "sigma": 0.6, "min": 12, "max": 5000, "plateau": 1500},
                                   "dead": {"median": 350, "sigma": 0.7, "min": 60, "max": 6000, "plateau": 2500}},
                      "penalty_turnover": 10.0, "penalty_stoppage": 10.0, "penalty_restart": 6.0, "refine_frames": 10,
                      "emission_scale": 4.0, "boundary_gamma": 6.0}}


def decode_halves(p_home, p_dead, g, boundary, forced_dead=None, cfg=CONFIG):
    """Labels (H, A, D; 3 = X outside g) from the team probability, the dead-ball probability and the boundary
    probability, each run of g decoded on its own."""
    p, pd = np.asarray(p_home, np.float64), np.asarray(p_dead, np.float64)
    P = np.stack([(1 - pd) * p, (1 - pd) * (1 - p), pd], 1)
    out = np.full(len(g), 3, np.int8)
    e = np.diff(np.r_[0, np.asarray(g).astype(np.int8), 0])
    for a, c in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        out[a:c] = decode_run(emission_logprob(P[a:c], cfg), cfg, forced_dead=None if forced_dead is None else forced_dead[a:c],
                              boundary=np.asarray(boundary[a:c], np.float64))
    return out
