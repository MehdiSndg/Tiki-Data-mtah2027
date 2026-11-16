"""Whistle timing (Gün 5, Ek N.13): move play -> dead boundaries to the collective deceleration onset.

On BAR-ATH most stoppages are fouls / whistles with no ball-out evidence (ball crosses a line within
+-15 frames of only 7/39 true dead starts), and the decoder's play -> dead boundary is off by a
median 14-15 frames, p75 47. The players' 80th-percentile speed, however, collapses at the
whistle (median over 39 true starts: 3.4 m/s at -25 frames, 2.4 at 0, 1.8 at +25).

For each decoded play -> dead boundary t: in [t - back, t + fwd] find the frame c where the
smoothed p80 speed crosses the midpoint between its level before (mean over [c - pre, c)) and
after (mean over [c + gap, c + gap + post)) with the strongest drop; move the boundary to c + shift
only if the drop is >= min_drop m/s and |c + shift - t| >= min_move frames, keeping the
preceding possession >= min_alive frames. Players only (no ball, no labels). Off unless
decoder.whistle.enabled; parameters are calibrated on DFL, never on the blind blocks.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..constants import A, D, H

DEFAULTS = dict(enabled=False, back=60, fwd=30, pre=25, gap=15, post=40, smooth=13, min_drop=0.8,
                min_move=15, shift=0, min_alive=12)


def params(cfg):
    return dict(DEFAULTS, **((cfg.get("decoder") or {}).get("whistle") or {}))


def collective_p80(players, fps):
    v = np.linalg.norm(np.gradient(np.asarray(players, np.float64), 1.0 / fps, axis=0), axis=-1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p80 = np.nanpercentile(v, 80, axis=1)
    return np.nan_to_num(p80, nan=np.nanmedian(p80))


def refine_dead_starts(labels, runs, p80, wp, info=None):
    lab = labels.copy()
    s = uniform_filter1d(p80, int(wp["smooth"]))
    cs = np.r_[0.0, np.cumsum(s)]

    def mean(a, b):
        a, b = max(a, 0), min(b, len(s))
        return (cs[b] - cs[a]) / max(b - a, 1)

    moved = 0
    for a, b in runs:
        t = a + 1
        while t < b:
            if lab[t] == D and lab[t - 1] in (H, A):
                prev = int(lab[t - 1])
                s0 = t - 1
                while s0 > a and lab[s0 - 1] == prev:
                    s0 -= 1
                e_d = t
                while e_d < b and lab[e_d] == D:
                    e_d += 1
                best, arg = 0.0, None
                for c in range(max(s0 + wp["min_alive"], t - wp["back"]), min(e_d - 1, t + wp["fwd"]) + 1):
                    before = mean(c - wp["pre"], c)
                    after = mean(c + wp["gap"], c + wp["gap"] + wp["post"])
                    drop = before - after
                    if drop > best and (s[c - 1] >= (before + after) / 2 >= s[c] if c > 0 else False):
                        best, arg = drop, c
                if arg is not None and best >= wp["min_drop"]:
                    c = int(np.clip(arg + wp["shift"], s0 + wp["min_alive"], e_d - 1))
                    if abs(c - t) >= wp["min_move"]:
                        if c < t:
                            lab[c:t] = D
                        else:
                            lab[t:c] = prev
                        moved += 1
                        t = max(c, t) + 1
                        continue
            t += 1
    if info is not None:
        info["n_whistle_moved"] = moved
    return lab
