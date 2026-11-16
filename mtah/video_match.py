"""Match v2: our match with the video layer applied. (P3)

    from mtah.video_match import from_tactical_v2
    m = from_tactical_v2()                     # same Match format as mtah.match.from_tactical

  ball     mtah.balltrack.fuse_ball over the tactical ATD, a detector's best on-pitch box per frame
           (default: the pretrained YOLO pass in cache/yolo, class 32) and, only if asked, the
           broadcast ATD -- off by default: its ball, where the tactical view has no players
           around it, made the team call worse (eda_out/reports/P3.md); where the tactical ATD sees the
           ball nothing changes
  keepers  a keeper slot that is a GHOST (mtah.keepers) takes the position recovered from kit
           colour (cache/keeper_recovered.npz, analysis/p3g_keeper_recovery.py) when it belongs to
           a consistent tracklet (keeper_tracklets), or NaN; an OFFSCREEN slot (a stale position) becomes NaN; SEEN slots are untouched
Derived from caches that analysis scripts write; none of them reads a ground truth.
"""

import glob
import os

import numpy as np
import pandas as pd

from mtah.balltrack import fuse_ball
from mtah.broadcast import broadcast_ball_on_tactical
from mtah.data import GK_SLOTS, PITCH
from mtah.frames import load_frames
from mtah.keepers import GHOST, OFFSCREEN
from mtah.match import from_tactical

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_DIR = os.path.join(ROOT, "cache", "yolo")
RECOVERED = os.path.join(ROOT, "cache", "keeper_recovered.npz")
SPOT_R, MARGIN = 1.0, 1.0
KEEPER_SPEED = 10.0         # m/s; a keeper sprinting stays under it, a switch to another person does not
KEEPER_GAP = 25             # frames; a longer gap starts a new tracklet
KEEPER_MIN = 5              # frames; shorter tracklets are dropped
SPOTS = ((11.0, PITCH[1] / 2), (PITCH[0] - 11.0, PITCH[1] / 2), (PITCH[0] / 2, PITCH[1] / 2))


def detector_ball(hom, n, det_dir=YOLO_DIR, cls=32, conf_min=0.3):
    """Best on-pitch ball box per frame as normalised pitch (x, y), its confidence, and a spot flag."""
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(det_dir, "chunk_*.npz")))]
    b = pd.DataFrame({c: np.concatenate([p[c] for p in parts]) for c in ("frame", "cls", "conf", "x", "y")})
    b = b[(b.cls == cls) & (b.conf >= conf_min) & (b.frame < n)]
    q = np.einsum("nij,nj->ni", np.linalg.inv(hom[b.frame.values]), np.c_[b.x.values, b.y.values, np.ones(len(b))])
    X, Y = q[:, 0] / q[:, 2] * PITCH[0], q[:, 1] / q[:, 2] * PITCH[1]
    keep = (X > -MARGIN) & (X < PITCH[0] + MARGIN) & (Y > -MARGIN) & (Y < PITCH[1] + MARGIN)
    b = b.assign(X=X, Y=Y)[keep]
    b["spot"] = np.any([np.hypot(b.X - sx, b.Y - sy) < SPOT_R for sx, sy in SPOTS], axis=0)
    best = b.sort_values(["frame", "conf"], kind="stable").groupby("frame").tail(1)
    xy, conf, spot = np.full((n, 2), np.nan), np.zeros(n), np.zeros(n, bool)
    f = best.frame.values
    xy[f] = np.c_[best.X.values / PITCH[0], best.Y.values / PITCH[1]]
    conf[f], spot[f] = best.conf.values, best.spot.values
    return xy, conf, spot


def keeper_tracklets(frame, x_m, y_m):
    """Mask of recovered keeper points to keep: cut where the step is too fast or the gap too long,
    drop tracklets shorter than KEEPER_MIN frames."""
    n = len(frame)
    if n == 0:
        return np.zeros(0, bool)
    df = np.diff(frame)
    step = np.hypot(np.diff(x_m), np.diff(y_m))
    cut = (df > KEEPER_GAP) | (step > np.maximum(1.0, KEEPER_SPEED * df / 25))
    tid = np.r_[0, np.cumsum(cut)]
    size = np.bincount(tid)
    return size[tid] >= KEEPER_MIN


def from_tactical_v2(det_dir=YOLO_DIR, det_cls=32, use_broadcast=False, det=None):
    """det: a (xy, conf, spot) track as detector_ball returns it, instead of reading det_dir."""
    m = from_tactical()
    n = len(m["bx"])
    bro = np.c_[broadcast_ball_on_tactical()] if use_broadcast else np.full((n, 2), np.nan)
    det, conf, spot = det if det is not None else detector_ball(m["hom"] if "hom" in m else _hom(), n, det_dir, det_cls)
    fused = fuse_ball(np.c_[m["bx"], m["by"]].astype(float), bro.astype(float), det, conf, spot, PITCH)
    seen = ~np.isnan(m["bx"])
    bx = np.where(seen, m["bx"], fused["X"]).astype(np.float32)
    by = np.where(seen, m["by"], fused["Y"]).astype(np.float32)
    px, py = m["px"].copy(), m["py"].copy()
    df = load_frames()
    rec = np.load(RECOVERED) if os.path.exists(RECOVERED) else None
    for team, side in ((0, "H"), (1, "A")):
        k, st = GK_SLOTS[team], df[side + "_gk_status"].values
        bad = (st == GHOST) | (st == OFFSCREEN)
        px[bad, k], py[bad, k] = np.nan, np.nan
        if rec is not None:
            r = np.flatnonzero((rec["team"] == team) & (st[rec["frame"]] == GHOST))
            r = r[keeper_tracklets(rec["frame"][r], rec["x"][r] * PITCH[0], rec["y"][r] * PITCH[1])]
            px[rec["frame"][r], k], py[rec["frame"][r], k] = rec["x"][r], rec["y"][r]
    return dict(m, name="tactical_v2", bx=bx, by=by, px=px, py=py, ball_source=fused["source"])


def _hom():
    from mtah.data import load_tactical
    return load_tactical()["hom"]
