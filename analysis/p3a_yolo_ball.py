"""P3 groundwork: how does the pretrained-YOLO ball compare with the ATD ball? (GT-free)

    uv run python analysis/p3a_yolo_ball.py        # uses whatever cache/yolo chunks exist

Questions, in order (each answer is a precondition for the next):
  1. alignment   is the YOLO frame the ATD frame? The ATD ball projected to pixels is compared with
                 the nearest YOLO ball at time offsets -3..+3 frames; the median distance must be
                 smallest at 0 (OpenCV seeking in an mp4 can land a few frames off)
  2. agreement   when both see a ball, how often is the best YOLO ball the ATD ball, by confidence
  3. coverage    on frames where the ATD ball is missing, how often does YOLO see a ball at all,
                 split by whether the frame is in play or dead by the v0 vote (GT is not used)
  4. filtering   2 and 3 again with only boxes on the pitch (1 m margin): spare balls lie beside
                 the touchline near halfway, penalty and centre spots look like balls
  5. disagreements  best on-pitch box > 60 px from the ATD ball: "aerial-shaped" when YOLO is
                 straight above the ATD point (a ball in the air projects higher in the image than
                 its ground position, which is what the ATD gives), otherwise single-frame blips?
  6. spots       the painted penalty and centre spots are detected as balls: how often is a box
                 within SPOT_R of a spot false (ATD ball > 5 m away), by confidence, and how much of
                 the gap filling in 4 were spot boxes
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from mtah.ball import MISSING  # noqa: E402
from mtah.data import PITCH, load_tactical  # noqa: E402
from mtah.frames import load_frames  # noqa: E402

# Params
NEAR_PX = 20            # a YOLO ball this close to the projected ATD ball is "the same ball"
SAME_PX = 30            # looser match used once the boxes are filtered (ATD-to-image error grows with depth)
FAR_PX = 60
MARGIN = 1.0            # metres outside the lines still counted as on the pitch
SPOT_R = 1.0            # metres; the spot boxes fall within 0.5 m, the excess density ends by 1 m
SPOTS = ((11.0, PITCH[1] / 2), (PITCH[0] - 11.0, PITCH[1] / 2), (PITCH[0] / 2, PITCH[1] / 2))
CONFS = (0.10, 0.20, 0.30, 0.50)


def load_yolo():
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(ROOT, "cache", "yolo", "chunk_*.npz")))]
    cols = ("frame", "cls", "conf", "x", "y", "w", "h")
    d = pd.DataFrame({c: np.concatenate([p[c] for p in parts]) for c in cols})
    covered = np.unique(np.concatenate([np.arange(p["frame"].min(), p["frame"].max() + 1) for p in parts if len(p["frame"])]))
    return d, covered


def atd_ball_px(m):
    H = m["hom"]
    x, y = m["bx"], m["by"]
    d = H[:, 2, 0] * x + H[:, 2, 1] * y + H[:, 2, 2]
    return (H[:, 0, 0] * x + H[:, 0, 1] * y + H[:, 0, 2]) / d, (H[:, 1, 0] * x + H[:, 1, 1] * y + H[:, 1, 2]) / d


def to_pitch(balls, hom):
    Hi = np.linalg.inv(hom[balls.frame.values])
    p = np.einsum("nij,nj->ni", Hi, np.c_[balls.x.values, balls.y.values, np.ones(len(balls))])
    return p[:, 0] / p[:, 2] * PITCH[0], p[:, 1] / p[:, 2] * PITCH[1]


def on_spot(balls, hom):
    X, Y = to_pitch(balls, hom)
    return np.any([np.hypot(X - sx, Y - sy) < SPOT_R for sx, sy in SPOTS], axis=0)


def on_pitch(balls, hom):
    X, Y = to_pitch(balls, hom)
    return (X > -MARGIN) & (X < PITCH[0] + MARGIN) & (Y > -MARGIN) & (Y < PITCH[1] + MARGIN)


def nearest_ball_dist(balls, frames, u, v, conf_min):
    """For each frame f, the distance from (u[f], v[f]) to the nearest YOLO ball of conf >= conf_min."""
    b = balls[balls.conf >= conf_min]
    out = np.full(len(frames), np.inf)
    pos = {f: i for i, f in enumerate(frames)}
    for f, g in b.groupby("frame"):
        i = pos.get(f)
        if i is not None:
            out[i] = np.min(np.hypot(g.x.values - u[i], g.y.values - v[i]))
    return out


def main():
    m = load_tactical()
    df = load_frames()
    d, covered = load_yolo()
    balls = d[d.cls == 32]
    print("YOLO chunks cover {} frames ({}..{}); {:.2f} ball boxes and {:.1f} person boxes per frame".format(
        len(covered), covered.min(), covered.max(), len(balls) / len(covered), (d.cls == 0).sum() / len(covered)))
    u, v = atd_ball_px(m)
    seen = (df["ball_status"].values <= 1) & ~np.isnan(u)

    print("\n1. alignment: median px distance, ATD ball at frame f vs nearest YOLO ball (conf >= 0.3) at f + k")
    fr = covered[seen[covered]][::5]
    for k in range(-3, 4):
        ok = np.isin(fr + k, covered)
        dist = nearest_ball_dist(balls, fr[ok] + k, u[fr[ok]], v[fr[ok]], 0.3)
        print("   k={:+d}  median {:6.1f} px  within {} px {:5.1%}".format(k, np.median(dist[np.isfinite(dist)]), NEAR_PX, np.mean(dist < NEAR_PX)))

    print("\n2. agreement on frames where the ATD ball is seen")
    fs = covered[seen[covered]]
    best = balls.sort_values("conf").groupby("frame").tail(1).set_index("frame")
    for c in CONFS:
        bb = best[best.conf >= c]
        have = np.isin(fs, bb.index.values)
        f2 = fs[have]
        dist = np.hypot(bb.loc[f2, "x"].values - u[f2], bb.loc[f2, "y"].values - v[f2])
        print("   conf >= {:.2f}: YOLO has a ball on {:5.1%} of them; its best box is the ATD ball on {:5.1%}".format(
            c, have.mean(), np.mean(dist < NEAR_PX)))

    print("\n3. coverage where the ATD ball is missing (dead/alive from the v0 vote; GT not used)")
    v0 = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "v0_rules.csv")).label.values
    miss = covered[(df["ball_status"].values[covered] == MISSING)]
    for name, sel in (("v0 alive", np.isin(v0[miss], ["H", "A"])), ("v0 dead ", v0[miss] == "D")):
        f3 = miss[sel]
        row = []
        for c in CONFS:
            row.append("{:.2f}: {:5.1%}".format(c, np.isin(f3, best[best.conf >= c].index.values).mean()))
        print("   {} ({:6d} frames)  YOLO ball at conf >= {}".format(name, len(f3), "   ".join(row)))

    print("\n4. on-pitch boxes only")
    onp = balls[on_pitch(balls, m["hom"])]
    print("   {:.0%} of ball boxes are off the pitch".format(1 - len(onp) / len(balls)))
    bestp = onp.sort_values("conf").groupby("frame").tail(1).set_index("frame")
    for c in CONFS:
        bb = bestp[bestp.conf >= c]
        have = np.isin(fs, bb.index.values)
        f2 = fs[have]
        dist = np.hypot(bb.loc[f2, "x"].values - u[f2], bb.loc[f2, "y"].values - v[f2])
        cov_ = ["{} {:5.1%}".format(n.strip(), np.isin(miss[sel], bb.index.values).mean())
                for n, sel in (("alive", np.isin(v0[miss], ["H", "A"])), ("dead", v0[miss] == "D"))]
        print("   conf >= {:.2f}: ATD-seen frames with a ball {:5.1%}, best = ATD ball (< {} px) {:5.1%} | ATD-missing with a ball: {}".format(
            c, have.mean(), SAME_PX, np.mean(dist < SAME_PX), ", ".join(cov_)))

    print("\n5. disagreements (on-pitch, conf >= 0.30)")
    bb = bestp[bestp.conf >= 0.3]
    f2 = fs[np.isin(fs, bb.index.values)]
    dx, dy = bb.loc[f2, "x"].values - u[f2], bb.loc[f2, "y"].values - v[f2]
    dis = np.hypot(dx, dy) > FAR_PX
    aer = dis & (dy < 0) & (np.abs(dx) < 0.35 * np.abs(dy))
    spd = df["ball_speed"].values[f2]
    other = f2[dis & ~aer]
    lens = np.array([len(r) for r in np.split(other, np.flatnonzero(np.diff(other) > 1) + 1)]) if len(other) else np.array([0])
    print("   {:.1%} of frames disagree; aerial-shaped {:.0%} of those (ATD ball speed median {:.1f} m/s, "
          "agreeing frames {:.1f}); the rest are runs of 1 frame in {:.0%} of cases".format(
              dis.mean(), aer.sum() / max(dis.sum(), 1), np.nanmedian(spd[aer]), np.nanmedian(spd[~dis]), np.mean(lens == 1)))

    print("\n6. painted spots (on-pitch boxes within {} m of a penalty or the centre spot)".format(SPOT_R))
    spot = on_spot(onp, m["hom"])
    X, Y = to_pitch(onp, m["hom"])
    f = onp.frame.values
    atd = np.hypot(X - m["bx"][f] * PITCH[0], Y - m["by"][f] * PITCH[1])
    known = spot & ~np.isnan(atd)
    for c in CONFS:
        k = known & (onp.conf.values >= c)
        print("   conf >= {:.2f}: {:6d} spot boxes on ATD-seen frames, false (ATD ball > 5 m away) {:5.1%}".format(c, k.sum(), np.mean(atd[k] > 5)))
    alive_miss = miss[np.isin(v0[miss], ["H", "A"])]
    for c in (0.2, 0.3):
        k = onp[onp.conf.values >= c]
        bk = k.sort_values("conf").groupby("frame").tail(1)
        filled = np.isin(alive_miss, bk.frame.values).mean()
        bk2 = k[~spot[onp.conf.values >= c]].sort_values("conf").groupby("frame").tail(1)
        print("   conf >= {:.2f}: in-play ATD gaps filled {:5.1%} with spot boxes, {:5.1%} without".format(
            c, filled, np.isin(alive_miss, bk2.frame.values).mean()))


if __name__ == "__main__":
    main()
