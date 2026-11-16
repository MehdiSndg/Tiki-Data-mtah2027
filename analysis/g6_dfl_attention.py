"""Do players move towards the ball while it is in play, and stop doing so when it is dead?

    uv run python analysis/g6_dfl_attention.py

The user's idea (22 Oct): players look at the ball in play and look anywhere once it is out. Gaze is
not visible on the tactical camera (players are 30-60 px tall); the direction they MOVE is in the
tracking. Measured on DFL (external; its BallStatus says alive or dead), so the public ground truth
plays no part:

  alignment  per frame, the mean over moving players (speed > MOVING m/s) of the cosine between
             their velocity (over 0.2 s) and the direction to the ball; +1 everyone runs at the ball
  moving     share of players moving at all

Then, as a dead-ball signal: the balanced accuracy of "alignment (2 s median) below a threshold"
against BallStatus, next to the player-speed signal already in use (1.4 m/s).
"""

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from mtah.external import DFL, _dfl_raw  # noqa: E402

# Params
FPS = 25
LAG = 5                 # frames for the velocity
MOVING = 1.0            # m/s
SMOOTH = 2 * FPS


def frame_features(d):
    L, W = d["pitch"]
    X, Y = d["px"].astype(float) * L, d["py"].astype(float) * W
    bX, bY = d["bx"].astype(float) * L, d["by"].astype(float) * W
    vx = np.full_like(X, np.nan)
    vy = vx.copy()
    vx[LAG:], vy[LAG:] = (X[LAG:] - X[:-LAG]) * FPS / LAG, (Y[LAG:] - Y[:-LAG]) * FPS / LAG
    sp = np.hypot(vx, vy)
    tx, ty = bX[:, None] - X, bY[:, None] - Y
    tn = np.hypot(tx, ty)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cos = (vx * tx + vy * ty) / (sp * tn)
        mov = sp > MOVING
        align = np.nanmean(np.where(mov & (tn > 2), cos, np.nan), axis=1)
        moving = np.sum(mov, axis=1) / np.sum(~np.isnan(sp), axis=1)
        speed = np.nanmedian(sp, axis=1)
    return align, moving, speed


def balanced(score, dead, thresholds, below=True):
    best = (0, None)
    for t in thresholds:
        pred = score < t if below else score > t
        ba = (np.mean(pred[dead]) + np.mean(~pred[~dead])) / 2
        best = max(best, (ba, t))
    return best


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    A, M, S, dead = [], [], [], []
    for mid in ids:
        d = _dfl_raw(mid)
        d = dict(d, pitch=tuple(d["pitch"]))
        a, mv, sp = frame_features(d)
        g = d["half"] > 0
        for arr, out in ((a, A), (mv, M), (sp, S)):
            out.append(pd.Series(np.where(g, arr, np.nan)).rolling(SMOOTH, center=True, min_periods=10).median().values[g])
        dead.append(d["raw"][g] == 2)
    A, M, S, dead = map(np.concatenate, (A, M, S, dead))
    ok = ~np.isnan(A) & ~np.isnan(S)
    A, M, S, dead = A[ok], M[ok], S[ok], dead[ok]
    print("DFL, {} matches, {} frames (2 s medians)".format(len(ids), len(A)))
    for name, x in (("alignment to ball", A), ("share moving", M), ("median speed m/s", S)):
        print("   {:<18} alive: median {:+.2f} (p25 {:+.2f}, p75 {:+.2f}) | dead: median {:+.2f} (p25 {:+.2f}, p75 {:+.2f})".format(
            name, *np.percentile(x[~dead], [50, 25, 75]), *np.percentile(x[dead], [50, 25, 75])))
    ba_a = balanced(A, dead, np.arange(-0.2, 0.61, 0.05))
    ba_s = balanced(S, dead, np.arange(0.8, 2.01, 0.1))
    print("dead-ball signal, balanced accuracy: alignment < {:.2f}: {:.3f} | speed < {:.1f}: {:.3f}".format(ba_a[1], ba_a[0], ba_s[1], ba_s[0]))
    # complementary? both votes
    for name, pred in (("speed only", S < ba_s[1]), ("alignment only", A < ba_a[1]),
                       ("speed AND alignment", (S < ba_s[1]) & (A < ba_a[1])),
                       ("speed OR alignment", (S < ba_s[1]) | (A < ba_a[1]))):
        rec, spec = np.mean(pred[dead]), np.mean(~pred[~dead])
        print("   {:<20} dead recall {:.1%}  alive specificity {:.1%}  balanced {:.3f}".format(name, rec, spec, (rec + spec) / 2))


if __name__ == "__main__":
    main()
