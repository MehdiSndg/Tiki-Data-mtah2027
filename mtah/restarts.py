"""Restart evidence from Smart Tagging's set pieces: the team taking a corner, goal kick or free kick
has the ball when play resumes (CHALLENGE.md: possession changes only when the other team gains
controlled possession). Smart Tagging is Metrica's automatic tagging of the provided video, a
permitted input; its set pieces carry the taking team (checked against the corner's end of the
pitch: 9 of the 10 corners with the ball clearly at a corner). Throw-ins carry no team and are left out.

    from mtah.restarts import restart_evidence
    E = restart_evidence(events, alive)      # (n,) log-odds, + for home

From the first alive frame at or after the set piece's start, for HOLD frames: +W for home, -W for away.
W x HOLD far exceeds the Viterbi switch cost, so the restart team is fixed at the restart and the
model decides what happens after.
"""

import numpy as np

FPS = 25
HOLD = 2 * FPS
W = 3.0
TEAMS = {"Barcelona": 1, "Atletic Club": -1}
TYPES = {"corner kick", "goal kick", "free kick"}


def restart_evidence(events, alive, strict=False):
    """strict (v17): fire only on the first frame where a dead (not alive) stretch ends inside the window, and not at
    all without one. Smart Tagging's set-piece phase starts 5-7 s before the restart, so when the dead ball is caught
    late or missed, the first alive frame of the window is still the previous possession (3 of 26 set pieces, v16)."""
    n = len(alive)
    E = np.zeros(n)
    for e in events:
        if e["code"] != "SET PIECES" or (e["tags"].get("Type") or "").strip().lower() not in TYPES:
            continue
        side = TEAMS.get(e["tags"].get("Team"))
        if side is None:
            continue
        a = max(0, e["start_frame"])
        win = alive[a:min(n, e["end_frame"] + 5 * FPS)]
        if strict:
            before = np.r_[alive[a - 1] if a > 0 else True, win[:-1]]
            nxt = np.flatnonzero(win & ~before)
        else:
            nxt = np.flatnonzero(win)
        if not len(nxt):
            continue
        s = a + int(nxt[0])
        E[s:min(n, s + HOLD)] = side * W
    return E
