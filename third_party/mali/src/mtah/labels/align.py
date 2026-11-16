"""Align external possession labels with the BAR-ATH ground-truth semantics.

DFL possession flips on single touches and deflections (spell median 3.7 s, 10-17% under 1 s);
Metrica's corrected tracking (our GT) follows controlled possession. An A-B-A pattern whose B
spell is at most `max_frames` long becomes A-A-A. Dead-ball segments are never touched.

Adım 2 used max_frames = 11 (shorter than the shortest public spell). Gün 2 (report 3, 1.3):
max_frames = 35 (1.4 s) with a control check -- the B spell is KEPT when the tracking shows real
control by B: two different B players touch the ball inside it (a completed pass) or one B player
stays within `contact_m` of the ball for at least `dribble_frames` (> 1 s carry). Contacts are
measured on the CLEAN track (data.ball), so the check needs the TrackingData.
"""

import numpy as np

from ..constants import A, H


def b_controlled(data, a, b, team_code, contact_m=1.0, dribble_frames=25):
    """True when team `team_code` (H=0 -> team 0) visibly controls the ball in frames [a, b)."""
    if data is None:
        return False
    side = 0 if team_code == H else 1
    P = data.players[a:b][:, data.team == side]
    ball = data.ball[a:b]
    d = np.hypot(P[..., 0] - ball[:, None, 0], P[..., 1] - ball[:, None, 1])
    touch = np.nan_to_num(d, nan=99.0) < contact_m              # (k, n_side)
    if touch.any(0).sum() >= 2:
        return True                                             # two different players: a pass
    return bool(touch.sum(0).max(initial=0) >= dribble_frames)


def controlled_possession(labels, runs, data, max_frames=75, min_frames=19, before_dead=True,
                          contact_m=1.0, dribble_frames=25):
    """Gün 3: DFL touch-level labels -> Metrica-like CONTROLLED possession.

    A spell of team B that follows team A is an uncontrolled touch (deflection, block, failed
    interception) and is merged into A when it is
      - A-B-A (A gets the ball back) or, with before_dead, A-B-D (B's touch sends the ball dead:
        Metrica keeps A until the stoppage), and
      - shorter than min_frames (the shortest public GT possession is 19 frames), or at most
        max_frames long without visible control by B (b_controlled: a completed B pass or a
        >= dribble_frames carry).
    D-B-A (B restarts the game) is a set piece, i.e. control, and is never merged. Dead-ball
    segments are never changed. Parameters are calibrated on Metrica's event-derived labels
    (possessions per alive hour, median length), never on the BAR GT."""
    out = np.asarray(labels).copy()
    for a, b in runs:
        kept = set()
        while True:
            w = out[a:b]
            breaks = np.flatnonzero(w[1:] != w[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [len(w)]))
            kinds = w[bounds[:-1]]
            lengths = np.diff(bounds)
            hit = False
            for i in range(1, len(kinds) - 1):
                s, e = a + bounds[i], a + bounds[i + 1]
                prev, cur, nxt = kinds[i - 1], kinds[i], kinds[i + 1]
                if cur not in (H, A) or prev not in (H, A) or prev == cur or (s, e) in kept:
                    continue
                if not (nxt == prev or (before_dead and nxt not in (H, A))):
                    continue
                if lengths[i] >= min_frames:
                    if lengths[i] > max_frames or b_controlled(data, s, e, cur, contact_m, dribble_frames):
                        kept.add((s, e))
                        continue
                w[bounds[i]:bounds[i + 1]] = prev
                hit = True
            if not hit:
                break
    return out


def suppress_short_returns(labels, runs, max_frames=11, data=None, check_control=False,
                           contact_m=1.0, dribble_frames=25):
    out = np.asarray(labels).copy()
    for a, b in runs:
        kept = set()
        while True:
            w = out[a:b]
            breaks = np.flatnonzero(w[1:] != w[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [len(w)]))
            kinds = w[bounds[:-1]]
            lengths = np.diff(bounds)
            hit = False
            for i in range(1, len(kinds) - 1):
                s, e = a + bounds[i], a + bounds[i + 1]
                if (kinds[i] in (H, A) and lengths[i] <= max_frames and kinds[i - 1] == kinds[i + 1]
                        and kinds[i - 1] in (H, A) and kinds[i - 1] != kinds[i] and (s, e) not in kept):
                    if check_control and b_controlled(data, s, e, kinds[i], contact_m, dribble_frames):
                        kept.add((s, e))
                        continue
                    w[bounds[i]:bounds[i + 1]] = kinds[i - 1]
                    hit = True
            if not hit:
                break
    return out
