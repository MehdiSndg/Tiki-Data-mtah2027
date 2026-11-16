"""Dead-ball context for the team nets (P9ai). 76% of the 0-2 s possessions border a dead ball, and at a line stoppage
the restart goes to the opponent of the last toucher 78% of the time (research report "Yapısal sıçrama adayları"),
but a stoppage lasts a median 22 s while the nets see 30 s windows, and the decoder restarts after every dead ball.
Per frame, from the predicted dead-ball blocks and the tracking only:

    0  seconds until the next dead block starts (CAP_S if none in the half)
    1  seconds since the previous dead block ended
    2  the team that restarts the next block: +1 home, -1 away, 0 unknown (nearest team to the first ball sighting
       after the block, within RESTART_S)
    3  the team that restarted the previous block
    4  metres from the last ball sighting before the next block to the nearest line (CAP_M if unknown)

All zero on dead frames.

    from mtah.deadctx import dead_context
    C = dead_context(m, dead)          # (n, 5) float32
"""

import numpy as np

FPS = 25
CAP_S = 60.0
CAP_M = 30.0
RESTART_S = 3.0


def _blocks(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def restart_team(m, end):
    """+1 / -1 / 0: the team of the player nearest the first ball sighting in [end, end + RESTART_S)."""
    L, W = m["pitch"]
    hi = min(len(m["bx"]), end + int(RESTART_S * FPS))
    seen = np.flatnonzero(np.isfinite(m["bx"][end:hi]))
    if not len(seen):
        return 0
    i = end + int(seen[0])
    d = np.hypot((m["px"][i] - m["bx"][i]) * L, (m["py"][i] - m["by"][i]) * W)
    if not np.isfinite(d).any():
        return 0
    return 1 if int(m["slot_team"][int(np.nanargmin(d))]) == 0 else -1


def _line_m(m, a):
    L, W = m["pitch"]
    seen = np.flatnonzero(np.isfinite(m["bx"][:a]))
    if not len(seen):
        return CAP_M
    x, y = m["bx"][seen[-1]] * L, m["by"][seen[-1]] * W
    return float(min(CAP_M, x, L - x, y, W - y))


def dead_context(m, dead):
    n = len(m["bx"])
    out = np.zeros((n, 5), np.float32)
    half = np.asarray(m["half"])
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        a0, b0 = idx[0], idx[-1] + 1
        bl = [(a0 + a, a0 + b) for a, b in _blocks(dead[a0:b0])]
        teams = [restart_team(m, b) for _, b in bl]
        lines = [_line_m(m, a) for a, _ in bl]
        starts = np.array([a for a, _ in bl])
        ends = np.array([b for _, b in bl])
        for t in range(a0, b0):
            if dead[t]:
                continue
            k = np.searchsorted(starts, t, side="right")          # next block index
            if k < len(bl):
                out[t, 0] = min(CAP_S, (starts[k] - t) / FPS)
                out[t, 2], out[t, 4] = teams[k], lines[k]
            else:
                out[t, 0], out[t, 4] = CAP_S, CAP_M
            j = np.searchsorted(ends, t, side="right") - 1          # previous block index
            if j >= 0:
                out[t, 1], out[t, 3] = min(CAP_S, (t - ends[j]) / FPS), teams[j]
            else:
                out[t, 1] = CAP_S
    return out
