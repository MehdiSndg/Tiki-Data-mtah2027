"""DFL's raw player tracks jitter frame to frame: per-frame acceleration median 6.3 m/s² against our ATD's 0.8 and
Metrica's 1.4, while speeds over 5 frames agree (1.4 m/s everywhere; 13 Nov label-free scan). Our degradation does not
touch player tracks, so the jitter reaches every per-frame player feature in training. dejitter(m, frames) replaces
each player's positions by a centred moving average inside each unbroken piece of track (no smoothing across a gap or
the half break; the window shrinks symmetrically at a piece's ends, so a constant-velocity track is unchanged).
Ball, labels and everything else are left alone. External data only (recipe E, P9et).

    from mtah.dejitter import dejitter, player_acc_median
"""

import numpy as np

FPS = 25


def _pieces(ok, half):
    """(start, end) of the runs where ok holds and the half does not change."""
    edge = np.r_[True, (half[1:] != half[:-1]) | ~ok[:-1]] & ok
    starts = np.flatnonzero(edge)
    out = []
    for a in starts:
        b = a + 1
        while b < len(ok) and ok[b] and half[b] == half[a]:
            b += 1
        out.append((a, b))
    return out


def _smooth(x, a, b, h):
    seg = x[a:b].astype(np.float64)
    n = len(seg)
    c = np.r_[0.0, np.cumsum(seg)]
    i = np.arange(n)
    w = np.minimum(h, np.minimum(i, n - 1 - i))
    return (c[i + w + 1] - c[i - w]) / (2 * w + 1)


def dejitter(m, frames):
    out = dict(m)
    h = frames // 2
    half = np.asarray(m["half"])
    px, py = np.array(m["px"], np.float64), np.array(m["py"], np.float64)
    for j in range(px.shape[1]):
        ok = np.isfinite(px[:, j]) & np.isfinite(py[:, j])
        for a, b in _pieces(ok, half):
            px[a:b, j] = _smooth(px[:, j], a, b, h)
            py[a:b, j] = _smooth(py[:, j], a, b, h)
    out["px"], out["py"] = px.astype(np.asarray(m["px"]).dtype), py.astype(np.asarray(m["py"]).dtype)
    return out


def player_acc_median(m, fps=FPS):
    """Median per-frame acceleration (m/s²) of every player over three consecutive tracked frames of one half (> 0)."""
    L, W = m["pitch"]
    x, y = np.asarray(m["px"], np.float64) * L, np.asarray(m["py"], np.float64) * W
    half = np.asarray(m["half"])
    same = (half[2:] == half[1:-1]) & (half[1:-1] == half[:-2]) & (half[1:-1] > 0)
    ax = x[2:] - 2 * x[1:-1] + x[:-2]
    ay = y[2:] - 2 * y[1:-1] + y[:-2]
    acc = np.hypot(ax, ay)[same] * fps ** 2
    acc = acc[np.isfinite(acc)]
    return float(np.median(acc)) if len(acc) else float("nan")
