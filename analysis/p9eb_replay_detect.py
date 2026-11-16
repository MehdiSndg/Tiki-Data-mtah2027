"""P9eb: a label-free replay detector from the broadcast (the friend's v3: replays / close-ups as dead-ball evidence).
On the synchronised timeline (cache/broadcast_on_tactical.npz) a broadcast frame showing the pitch live agrees with the
tactical ATD of the same moment; a replay shows an earlier moment, so its players (mapped by the broadcast homography)
do not match the tactical players. Per tactical frame with a valid broadcast homography: median over broadcast players
of the distance to the nearest tactical player (m), and the ball distance when both have a ball. Reports the
distribution, the share of in-play frames flagged (mismatch > 4 m sustained >= 2 s), and their overlap with v26's dead
ball and the video models' dead calls (no labels). Writes cache/p9eb_replay.npz.

    .venv/bin/python analysis/p9eb_replay_detect.py
"""
import os
import sys

import numpy as np
from scipy.ndimage import median_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
from games import asset_paths, source_frames  # noqa: E402
from mtah.match import from_tactical  # noqa: E402

FPS = 25


def broadcast_players():
    n = source_frames("broadcast")
    P = np.full((n, 46, 2), np.nan, np.float32)
    B = np.full((n, 2), np.nan, np.float32)
    with open(asset_paths(source="broadcast")["atd_raw"]) as f:
        for line in f:
            a, pl, b = line.rstrip("\n").split(":")
            i = int(a) - 1
            for j, s in enumerate(pl.split(";")[:46]):
                if s[0] != "N":
                    x, y, _ = s.split(",")
                    P[i, j] = float(x), float(y)
            if b[0] != "N":
                x, y, _ = b.split(",")
                B[i] = float(x), float(y)
    return P, B


def main():
    m = from_tactical()
    L, W = m["pitch"]
    g = m["half"] > 0
    s = np.load(os.path.join(ROOT, "cache", "broadcast_on_tactical.npz"))
    bf, present, valid = s["b_frame"], s["b_present"], s["b_valid"]
    P, B = broadcast_players()
    n = len(g)
    mis = np.full(n, np.nan)
    bdist = np.full(n, np.nan)
    tp = np.stack([m["px"] * L, m["py"] * W], -1)
    for t in np.flatnonzero(g & present & valid & (bf >= 0) & (bf < len(P))):
        q = P[bf[t]] * np.array([L, W])
        q = q[np.isfinite(q[:, 0])]
        r = tp[t][np.isfinite(tp[t][:, 0])]
        if len(q) >= 3 and len(r) >= 3:
            d = np.linalg.norm(q[:, None] - r[None], axis=-1).min(1)
            mis[t] = np.median(d)
        if np.isfinite(B[bf[t], 0]) and np.isfinite(m["bx"][t]):
            bdist[t] = np.hypot((B[bf[t], 0] - m["bx"][t]) * L, (B[bf[t], 1] - m["by"][t]) * W)
    ok = np.isfinite(mis)
    print("frames with a usable comparison: {:.3f} of in-play".format(ok[g].mean()))
    print("player mismatch (m) percentiles 50/75/90/95/99:", np.round(np.nanpercentile(mis[ok], [50, 75, 90, 95, 99]), 2))
    print("ball distance (m) percentiles 50/75/90/95/99:", np.round(np.nanpercentile(bdist[np.isfinite(bdist)], [50, 75, 90, 95, 99]), 2))
    sm = median_filter(np.nan_to_num(mis, nan=0.0), size=2 * FPS + 1)
    flag = g & (sm > 4.0)
    d = np.diff(np.r_[0, flag.astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if z - a < 2 * FPS:
            flag[a:z] = False
    import pandas as pd
    v26 = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "v26_chain.csv")).label.values == "D"
    vq = np.load(os.path.join(ROOT, "cache", "barath_dense_avg5.npz"))["q_dead"].astype(np.float64) > 0.5
    runs = int(np.sum(np.diff(np.r_[0, flag.astype(np.int8)]) == 1))
    print("flagged (mismatch > 4 m for >= 2 s): {:.3f} of in-play frames, {} stretches".format(flag[g].mean(), runs))
    print("of the flagged frames: v26 dead {:.3f}, video dead {:.3f}; of v26's live frames flagged: {:.4f} ({} frames)".format(
        v26[flag].mean(), vq[flag].mean(), flag[g & ~v26].mean(), int(np.sum(flag & g & ~v26))))
    np.savez_compressed(os.path.join(ROOT, "cache", "p9eb_replay.npz"), mis=mis.astype(np.float32), bdist=bdist.astype(np.float32), flag=flag)


if __name__ == "__main__":
    main()
