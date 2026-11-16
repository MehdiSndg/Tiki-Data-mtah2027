"""EDA section C: how good is the automatic tracking, over the whole match? No ground truth used.

    uv run python analysis/c_tracking_quality.py

Prints every number it computes; the findings are written up in notes/VERI.md.
Every statistic is over GRADED frames unless it says otherwise.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from mtah.data import FPS, GK_SLOTS, PITCH, load_tactical, project  # noqa: E402

IMG_W, IMG_H = 1920, 1080


def runs(mask):
    """[(start, stop)] maximal runs of True."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def pct(a, qs=(50, 75, 90, 99)):
    return "  ".join("p{} {:.2f}".format(q, np.percentile(a, q)) for q in qs)


def main():
    m = load_tactical()
    g = m["graded"]
    n_g = int(g.sum())
    bx, by = m["bx"], m["by"]
    seen = ~np.isnan(bx)

    print("== ball")
    print("visible on {:.1%} of all frames, {:.1%} of graded".format(seen.mean(), seen[g].mean()))
    gaps = [(a, b) for a, b in runs(~seen & g)]
    lens = np.array([b - a for a, b in gaps]) / FPS
    print("graded gaps: {}  length s: {}  max {:.1f}".format(len(gaps), pct(lens), lens.max()))
    for t in (0.2, 1, 3, 10):
        share = lens[lens > t].sum() * FPS / (~seen & g).sum()
        print("   share of missing graded frames in gaps > {:>4} s: {:.1%}".format(t, share))

    step = np.full(len(bx), np.nan)
    step[1:] = np.hypot(np.diff(bx) * PITCH[0], np.diff(by) * PITCH[1]) * FPS
    ok = g & ~np.isnan(step)
    print("frame-to-frame ball speed (m/s) on graded frames: {}".format(pct(step[ok], (50, 90, 99, 99.9))))
    for v in (40, 60, 100):
        print("   steps faster than {} m/s: {} ({:.3%})".format(v, int((step[ok] > v).sum()), (step[ok] > v).mean()))
    out = seen & g & ((bx < 0) | (bx > 1) | (by < 0) | (by > 1))
    print("ball outside the pitch rectangle: {:.2%} of graded frames with a ball".format(out.sum() / (seen & g).sum()))
    bs = m["bs"]
    print("ball speed column present on {:.1%} of frames that have a ball position".format((~np.isnan(bs[seen])).mean()))

    print("\n== players")
    present = ~np.isnan(m["px"])
    for team in (0, 1):
        cols = np.flatnonzero(m["slot_team"] == team)
        outfield = [c for c in cols if c != GK_SLOTS[team]]
        cnt = present[:, outfield].sum(1)[g]
        print("{} outfield objects per graded frame: mean {:.1f}  p5 {:.0f}  p50 {:.0f}  p95 {:.0f}  "
              "frames >10: {:.2%}  frames <7: {:.1%}".format("HA"[team], cnt.mean(), *np.percentile(cnt, [5, 50, 95]),
                                                        (cnt > 10).mean(), (cnt < 7).mean()))
    seglen = []
    for s in range(present.shape[1]):
        if s in GK_SLOTS.values():
            continue
        seglen += [(b - a) / FPS for a, b in runs(present[:, s] & g)]
    seglen = np.array(seglen)
    print("outfield tracklets (continuous presence of a slot): {}  length s: {}".format(len(seglen), pct(seglen)))
    print("   tracklets shorter than 1 s: {:.1%}".format((seglen < 1).mean()))

    # implied speed of each slot vs its reported speed -> are identities switching (teleports)?
    dx = np.diff(m["px"], axis=0) * PITCH[0]
    dy = np.diff(m["py"], axis=0) * PITCH[1]
    v = np.hypot(dx, dy) * FPS
    vg = v[g[1:] & g[:-1]]
    vg = vg[~np.isnan(vg)]
    print("player frame-to-frame speed m/s: {}   steps > 12 m/s (implausible): {:.3%}".format(
        pct(vg, (50, 99, 99.9)), (vg > 12).mean()))

    print("\n== camera: does the view follow the ball?")
    inv = np.linalg.inv(m["hom"])
    cx, cy = project(inv.transpose(1, 2, 0), IMG_W / 2.0, IMG_H / 2.0) if False else (None, None)
    centre = np.array([project(inv[f], IMG_W / 2.0, IMG_H / 2.0) for f in range(len(bx))])
    both = seen & g & np.all(np.isfinite(centre), axis=1)
    d_cam = np.hypot((centre[both, 0] - bx[both]) * PITCH[0], (centre[both, 1] - by[both]) * PITCH[1])
    d_mid = np.hypot((0.5 - bx[both]) * PITCH[0], (0.5 - by[both]) * PITCH[1])
    with np.errstate(all="ignore"):
        gx = np.nanmean(m["px"][both], axis=1)
        gy = np.nanmean(m["py"][both], axis=1)
    d_cen = np.hypot((gx - bx[both]) * PITCH[0], (gy - by[both]) * PITCH[1])
    print("distance to the ball (m) when the ball is visible:")
    print("   camera image centre : {}".format(pct(d_cam)))
    print("   players' centroid   : {}".format(pct(d_cen[~np.isnan(d_cen)])))
    print("   pitch centre        : {}".format(pct(d_mid)))
    # x only: the camera pans left-right; depth (y) is poorly constrained by a centre point
    print("   |x error| camera {:.1f} m vs centroid {:.1f} m (median)".format(
        np.median(np.abs(centre[both, 0] - bx[both]) * PITCH[0]), np.nanmedian(np.abs(gx - bx[both]) * PITCH[0])))

    # where does the ball go missing: inside the view, or outside it?
    last = np.full(len(bx), np.nan)
    idx = np.where(seen, np.arange(len(bx)), -1)
    idx = np.maximum.accumulate(idx)
    miss = ~seen & g & (idx >= 0)
    f = np.flatnonzero(miss)
    lx, ly = bx[idx[f]], by[idx[f]]
    inside = np.zeros(len(f), bool)
    for k, (fr, x, y) in enumerate(zip(f, lx, ly)):
        px, py = project(m["hom"][fr], x, y)
        inside[k] = 0 <= px < IMG_W and 0 <= py < IMG_H
    print("missing-ball graded frames whose LAST known ball position is inside the current view: {:.1%}".format(
        inside.mean()))
    np.savez_compressed(os.path.join(ROOT, "cache", "camera_centre.npz"), centre=centre.astype(np.float32))


if __name__ == "__main__":
    main()
