"""Post-processing after the HSMM (Gün 2, report 3 sections 4.2 / 5.2).

snap_boundaries  Kinematic boundary snapping: every decoded transition t is moved inside the
                 window [t - before, t + after] to
                   * live -> dead: the first frame the ball crosses a pitch line,
                   * any other transition (turnover, restart): the first detected kick/contact.
                 The kick detector is the per-match rate-calibrated `kin_kick` (12 kicks per
                 visible live minute), not a fixed |a| >= 4 m/s^2: on ATD 60% of frames exceed
                 4 m/s^2 (MASTER_PLAN Ek A), so a fixed threshold would fire everywhere.
                 Boundaries never cross their neighbours.
close_short_dead Short false D-gap closing: a dead segment shorter than `max_frames`, with the
                 same team on both sides and the ball never out of the pitch inside it, becomes
                 that team's possession.
Both work per run (graded half) and leave X frames alone.
"""

import numpy as np

from ..constants import A, D, H


def _segments(lab):
    b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.r_[0, b]
    stops = np.r_[b, len(lab)]
    return starts, stops


def oob_entries(ball, pitch, tol=0.0):
    """bool (T,): the ball is outside the pitch at t and was inside at t-1 (both visible)."""
    L, W = pitch
    x, y = ball[:, 0], ball[:, 1]
    out = (x < -tol) | (x > L + tol) | (y < -tol) | (y > W + tol)
    vis = np.isfinite(x)
    ent = np.zeros(len(x), bool)
    ent[1:] = out[1:] & vis[1:] & ~out[:-1] & vis[:-1]
    return ent


def snap_boundaries(labels, runs, kick, oob_in, before=20, after=15):
    out = labels.copy()
    for a, b in runs:
        lab = out[a:b].copy()
        starts, _ = _segments(lab)
        bounds = list(starts[1:])
        for i, t in enumerate(bounds):
            prev_b = bounds[i - 1] if i > 0 else 0
            next_b = bounds[i + 1] if i + 1 < len(bounds) else len(lab)
            lo, hi = max(prev_b + 1, t - before), min(next_b - 1, t + after)
            if hi <= lo:
                continue
            into_dead = lab[t] == D and lab[t - 1] in (H, A)
            sig = oob_in[a + lo:a + hi + 1] if into_dead else kick[a + lo:a + hi + 1]
            hits = np.flatnonzero(sig)
            if len(hits) == 0:
                continue
            new_t = lo + int(hits[0])
            left, right = lab[t - 1], lab[t]
            if new_t < t:
                lab[new_t:t] = right
            elif new_t > t:
                lab[t:new_t] = left
            bounds[i] = new_t
        out[a:b] = lab
    return out


def close_short_dead(labels, runs, ball, pitch, max_frames=125):
    out = labels.copy()
    L, W = pitch
    x, y = ball[:, 0], ball[:, 1]
    oob = np.isfinite(x) & ((x < 0) | (x > L) | (y < 0) | (y > W))
    for a, b in runs:
        lab = out[a:b]
        starts, stops = _segments(lab)
        for k in range(1, len(starts) - 1):
            s, e = starts[k], stops[k]
            if lab[s] != D or e - s >= max_frames:
                continue
            left, right = lab[starts[k - 1]], lab[starts[k + 1]]
            if left == right and left in (H, A) and not oob[a + s:a + e].any():
                lab[s:e] = left
        out[a:b] = lab
    return out


def apply(labels, runs, cfg, kick=None, ball=None, pitch=None):
    pc = cfg.get("post", {})
    out = labels
    if pc.get("close_short_dead") and ball is not None:
        out = close_short_dead(out, runs, ball, pitch, int(pc.get("short_dead_max_frames", 125)))
    if pc.get("snap") and kick is not None and ball is not None:
        out = snap_boundaries(out, runs, kick, oob_entries(ball, pitch),
                              int(pc.get("snap_before", 20)), int(pc.get("snap_after", 15)))
    return out


def report_postprocess(labels, runs, probs, ball, pitch, accel, near_home, near_away,
                       min_poss=14, max_false_d=95, pre=20, post_=15, min_kick_accel=4.0,
                       max_touch=2.2, steps=("close", "flicker", "snap")):
    """Report 3 section 5.2 / component 1, faithful port (per run):
      close   : D segments <= max_false_d frames between the same team, ball never out -> team
      flicker : segments shorter than min_poss frames merged into a neighbour (3 passes)
      snap    : turnover -> argmax over [t-pre, t+post] of 1.5*min(acc,20) - 2*d_dest + 4*p_dest
                (acc >= min_kick_accel, d_dest <= max_touch); live->dead -> first out-of-pitch
                frame; dead->live -> first frame with acc >= min_kick_accel and a destination
                player within max_touch. Margins keep every segment >= min_poss frames.
    accel: ball |a| (m/s^2, NaN when unknown); near_home/near_away: nearest player distance (m).
    min_kick_accel may be an array (per-frame threshold, e.g. the rate-calibrated kick level)."""
    out = labels.copy()
    L_, W_ = pitch
    x, y = ball[:, 0], ball[:, 1]
    oob = np.isfinite(x) & ((x < 0) | (x > L_) | (y < 0) | (y > W_))
    near = np.stack([near_home, near_away], axis=1)
    thr = np.broadcast_to(np.asarray(min_kick_accel, float), (len(labels),))
    for a, b in runs:
        lab = out[a:b].copy()
        if "close" in steps:
            st, en = _segments(lab)
            for i in range(1, len(st) - 1):
                if lab[st[i]] == D and lab[st[i - 1]] == lab[st[i + 1]] and lab[st[i - 1]] in (H, A) \
                        and en[i] - st[i] <= max_false_d and not oob[a + st[i]:a + en[i]].any():
                    lab[st[i]:en[i]] = lab[st[i - 1]]
        if "flicker" in steps:
            for _ in range(3):
                st, en = _segments(lab)
                changed = False
                for i in range(len(st)):
                    if en[i] - st[i] < min_poss and len(st) > 1:
                        if 0 < i < len(st) - 1 and lab[st[i - 1]] == lab[st[i + 1]]:
                            lab[st[i]:en[i]] = lab[st[i - 1]]
                        elif i > 0:
                            lab[st[i]:en[i]] = lab[st[i - 1]]
                        else:
                            lab[st[i]:en[i]] = lab[st[i + 1]]
                        changed = True
                if not changed:
                    break
        if "snap" in steps:
            st, en = _segments(lab)
            kinds = lab[st]
            new_st = list(st)
            for i in range(1, len(st)):
                t = st[i]
                lo = max(st[i - 1] + min_poss, t - pre)
                hi = min(en[i] - min_poss, t + post_)
                if lo >= hi:
                    continue
                prev_k, dest = kinds[i - 1], kinds[i]
                fa = a + np.arange(lo, hi + 1)
                acc = accel[fa]
                if prev_k in (H, A) and dest in (H, A):
                    dd = near[fa, dest]
                    ok = np.isfinite(dd) & np.isfinite(acc) & (dd <= max_touch) & (acc >= thr[fa])
                    if ok.any():
                        score = 1.5 * np.minimum(np.nan_to_num(acc), 20.0) - 2.0 * np.nan_to_num(dd) + 4.0 * probs[fa, dest]
                        score[~ok] = -np.inf
                        new_st[i] = lo + int(np.argmax(score))
                elif dest == D:
                    hit = np.flatnonzero(oob[fa])
                    if len(hit):
                        new_st[i] = lo + int(hit[0])
                elif prev_k == D and dest in (H, A):
                    dd = near[fa, dest]
                    ok = np.isfinite(dd) & np.isfinite(acc) & (dd <= max_touch) & (acc >= thr[fa])
                    hit = np.flatnonzero(ok)
                    if len(hit):
                        new_st[i] = lo + int(hit[0])
            res = lab.copy()
            bounds = new_st + [len(lab)]
            for i in range(len(st)):
                s0, e0 = bounds[i], bounds[i + 1]
                if e0 > s0:
                    res[s0:e0] = kinds[i]
            lab = res
        out[a:b] = lab
    return out
