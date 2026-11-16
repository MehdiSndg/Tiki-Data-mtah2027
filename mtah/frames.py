"""The P1 frame table: one row per video frame, every preprocessed input side by side.

    from mtah.frames import load_frames
    df = load_frames()          # cache/frames_v1.parquet, rebuilt when a source or module changes

Built only from the match bundle and the published half boundaries -- never from a ground truth
(tests/test_frames.py proves it by scrambling gt_public). Columns:

    period, half                       mtah.periods codes; half 1/2, 0 outside play
    ball_*                             mtah.ball.clean_ball, metres in the attack-direction frame
    H_n, A_n                           visible outfield slots per team (may exceed 10: duplicates)
    H_near_d, A_near_d                 metres from the ball to each team's nearest outfield slot
    H_gk_d, A_gk_d                     metres from the ball to each goalkeeper slot
    H_cx, A_cx                         mean x of each team's outfield slots (attack frame)
    med_speed                          median frame-to-frame speed of all outfield slots, m/s
    H_gk_X, A_gk_X                     goalkeeper x (attack frame)
    H_gk_status, A_gk_status           mtah.keepers: is the keeper slot backed by a YOLO person?
                                       (SEEN 0, GHOST 1, OFFSCREEN 2, ABSENT 3, UNKNOWN 4)
    view_xmin, view_xmax               pitch x range the camera shows, NORMALISED raw coordinates
                                       (camera geometry, not the attack frame)
    tag_H, tag_A, setpiece, setpiece_type, whistle   mtah.context
"""

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402
from games import asset_paths  # noqa: E402
from mtah.ball import INTERP, OBSERVED, clean_ball  # noqa: E402
from mtah.context import WHISTLE_CACHE, tag_features, whistle_feature  # noqa: E402
from mtah.data import CACHE_DIR, FPS, GK_SLOTS, PITCH, load_tactical  # noqa: E402
from mtah.keepers import YOLO_DIR, keeper_status, load_persons  # noqa: E402
from mtah.periods import attack_sign, half_of, periods  # noqa: E402
from mtah.players import team_arrays, view_box  # noqa: E402


def _nearest(xy, bX, bY):
    d = np.hypot(xy[..., 0] - bX[:, None], xy[..., 1] - bY[:, None])
    with np.errstate(all="ignore"):
        out = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    out[~np.isfinite(out)] = np.nan
    return out


def build_frames(version="v1"):
    meta = asset_paths()["atd_meta"]
    m = load_tactical()
    n = len(m["bx"])
    whistle = whistle_feature(n)
    code, _ = periods(m, meta, whistle=whistle)
    half = half_of(code)
    sign = attack_sign(meta)
    b = clean_ball(m["bx"], m["by"], code, half, sign)
    t = team_arrays(m, half, sign)

    usable = np.isin(b["status"], (OBSERVED, INTERP))
    bX = np.where(usable, b["X"], np.nan)
    bY = np.where(usable, b["Y"], np.nan)
    df = pd.DataFrame({"period": code, "half": half}, index=pd.RangeIndex(n, name="frame"))
    for key in ("X", "Y", "vx", "vy", "speed", "acc", "status", "off_pitch", "stationary", "gap_age", "gap_len"):
        df["ball_" + key] = b[key]
    # frames with no visible player or no usable ball give empty slices; NaN is the right answer
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for side in ("H", "A"):
            xy = t[side + "_xy"]
            df[side + "_n"] = t[side + "_n"]
            df[side + "_near_d"] = _nearest(xy, bX, bY).astype(np.float32)
            df[side + "_gk_d"] = np.hypot(t[side + "_gk"][:, 0] - bX, t[side + "_gk"][:, 1] - bY).astype(np.float32)
            df[side + "_cx"] = np.nanmean(xy[..., 0], axis=1).astype(np.float32)
            df[side + "_gk_X"] = t[side + "_gk"][:, 0]
        allxy = np.concatenate([t["H_xy"], t["A_xy"]], axis=1)
        step = np.full(allxy.shape[:2], np.nan, np.float32)
        step[1:] = np.hypot(*np.moveaxis(np.diff(allxy, axis=0), -1, 0)) * FPS
        df["med_speed"] = np.nanmedian(step, axis=1).astype(np.float32)
    persons, covered = load_persons()
    for side, team in (("H", 0), ("A", 1)):
        k = GK_SLOTS[team]
        df[side + "_gk_status"] = keeper_status(m["px"][:, k], m["py"][:, k], m["hom"], persons, covered, PITCH)
    vb = view_box(m["hom"])
    df["view_xmin"] = vb[:, 0].astype(np.float32)
    df["view_xmax"] = vb[:, 1].astype(np.float32)
    for key, val in tag_features(load_smart_tagging.load(), n).items():
        df[key] = val
    df["whistle"] = whistle
    df.attrs["version"] = version
    return df


def _sources():
    paths = list(asset_paths().values()) + [os.path.join(CACHE_DIR, "tactical_v1.npz"), WHISTLE_CACHE]
    paths += glob.glob(os.path.join(ROOT, "mtah", "*.py"))
    paths.append(os.path.join(ROOT, "games.py"))           # half boundaries
    paths += glob.glob(os.path.join(YOLO_DIR, "chunk_*.npz"))
    return [p for p in paths if os.path.exists(p) and not p.endswith(".mp4")]


def load_frames(version="v1"):
    """The cached frame table, rebuilt when any source file or mtah module is newer than it."""
    path = os.path.join(CACHE_DIR, "frames_{}.parquet".format(version))
    if not os.path.exists(path) or os.path.getmtime(path) < max(map(os.path.getmtime, _sources())):
        tmp = "{}.{}.tmp".format(path, os.getpid())       # swapped in whole: a parallel reader never sees half a file
        build_frames(version).to_parquet(tmp)
        os.replace(tmp, path)
    return pd.read_parquet(path)
