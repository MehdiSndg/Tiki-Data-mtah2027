"""One ball track from three sources: tactical ATD, broadcast ATD, a video detector. (P3)

    from mtah.balltrack import fuse_ball
    out = fuse_ball(tac_xy, bro_xy, det_xy, det_conf, spot, pitch)    # X, Y (0-1), source, conf

Order of trust, from the overnight measurements (notes/BULGULAR.md):
  tactical   when present it wins, also against the detector: when they disagree it is mostly an
             aerial ball, whose image sits above the ATD's ground point
  broadcast  fills a tactical gap (37% of in-play gaps); in disagreements YOLO sided with tactical
  detector   fills what is left when the box is reachable from the last fused position at a
             plausible speed; a box on a painted spot only when the ball was already there
  interp     gaps up to 1 s between two fused positions, linear
Everything is deterministic and frame-by-frame forward, then one interpolation pass.
"""

import numpy as np

# Params
FPS = 25
MAX_SPEED = 35.0            # m/s; the fastest shots; P1's outlier rule uses the same bound
DET_MIN = 0.3
SPOT_KEEP_M = 2.0           # a spot box continues a ball within this of it
MAX_INTERP = 1 * FPS
TACTICAL, BROADCAST, DETECTOR, INTERP, NONE = 0, 1, 2, 3, 4
CONF = {TACTICAL: 1.0, BROADCAST: 0.8, INTERP: 0.5}


def fuse_ball(tac_xy, bro_xy, det_xy, det_conf, spot, pitch):
    L, W = pitch
    n = len(tac_xy)
    X, Y = np.full(n, np.nan), np.full(n, np.nan)
    source = np.full(n, NONE, np.int8)
    conf = np.zeros(n)
    last, last_f = None, None
    for i in range(n):
        pick = None
        if not np.isnan(tac_xy[i, 0]):
            pick = (tac_xy[i], TACTICAL, CONF[TACTICAL])
        elif not np.isnan(bro_xy[i, 0]):
            pick = (bro_xy[i], BROADCAST, CONF[BROADCAST])
        elif not np.isnan(det_xy[i, 0]) and det_conf[i] >= DET_MIN:
            if last is not None:
                dist = np.hypot((det_xy[i, 0] - last[0]) * L, (det_xy[i, 1] - last[1]) * W)
                ok = dist <= SPOT_KEEP_M if spot[i] else dist <= MAX_SPEED * (i - last_f) / FPS
            else:
                ok = not spot[i]
            if ok:
                pick = (det_xy[i], DETECTOR, float(det_conf[i]))
        if pick is not None:
            (X[i], Y[i]), source[i], conf[i] = pick[0], pick[1], pick[2]
            last, last_f = (X[i], Y[i]), i
    known = np.flatnonzero(source != NONE)
    for a, b in zip(known[:-1], known[1:]):
        if 1 < b - a <= MAX_INTERP + 1:
            t = (np.arange(a + 1, b) - a) / (b - a)
            X[a + 1:b] = X[a] + (X[b] - X[a]) * t
            Y[a + 1:b] = Y[a] + (Y[b] - Y[a]) * t
            source[a + 1:b], conf[a + 1:b] = INTERP, CONF[INTERP]
    return {"X": X, "Y": Y, "source": source, "conf": conf}


# Online tracking of detector candidates (Tarashima et al., WASB, BMVC 2023, sec. 3.3): the next
# position is predicted from the last three picks (local motion model), candidates farther than a
# gate are dropped, the most confident of the rest is taken. Gate and restart are physical.
GATE_M = 1.0                 # m: slack around the prediction for one frame
LOST_FRAMES = 25             # frames (1 s) without a pick: the track is dropped
START_CONF = 0.3             # a new track starts only on a box this confident (DET_MIN)


def track_detections(frame, x, y, conf, n):
    """(X, Y, C) per frame 0..n-1 in metres from candidate boxes (frame, x, y in m, conf); NaN / 0 when none."""
    X, Y, C = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    order = np.argsort(frame, kind="stable")
    frame, x, y, conf = (np.asarray(a)[order] for a in (frame, x, y, conf))
    bounds = np.searchsorted(frame, np.arange(n + 1))
    picks = []                                   # (frame, x, y) of the current track
    for t in range(n):
        a, b = bounds[t], bounds[t + 1]
        if picks and t - picks[-1][0] > LOST_FRAMES:
            picks = []
        if a == b:
            continue
        cx, cy, cc = x[a:b], y[a:b], conf[a:b]
        if picks:
            gap = t - picks[-1][0]
            p = np.array(picks[-1][1:])
            if len(picks) >= 3 and picks[-1][0] - picks[-3][0] == 2:
                p2, p1, p0 = (np.array(q[1:]) for q in picks[-3:])
                acc = p0 - 2 * p1 + p2
                vel = p0 - p1 + acc
                p = p0 + (vel + acc / 2) * gap
            ok = np.hypot(cx - p[0], cy - p[1]) <= MAX_SPEED / FPS * gap + GATE_M
        else:
            ok = cc >= START_CONF
        if not ok.any():
            continue
        i = np.flatnonzero(ok)[np.argmax(cc[ok])]
        X[t], Y[t], C[t] = cx[i], cy[i], cc[i]
        picks = (picks + [(t, cx[i], cy[i])])[-3:]
    return X, Y, C
