"""P9cc: the virtual camera of P9bx on our match's tactical video (inference input only: ATD tracking and the provided
homography; no ground truth). Same rule as on SoccerTrack: a 960 x 540 window centred on the mean image position of the
two players nearest the ATD ball (feet projected through the frame's homography, raised by 20 px to the body centre as
GSR's box centres), held through missing ball frames and smoothed with a centred 1 s moving average. Also every
player's projected image position per frame (for matching YOLO boxes to ATD identities and teams).

    python analysis/p9cc_barath_vcam_plan.py      # -> cache/barath_vcam_plan.npz
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from mtah.data import load_tactical  # noqa: E402
from mtah.match import from_tactical  # noqa: E402

WIN_W, WIN_H, W, H, LIFT = 960, 540, 1920, 1080, 20.0
OUT = os.path.join(ROOT, "cache", "barath_vcam_plan.npz")


def project(hom, x, y):
    """normalised pitch (n, k) -> pixels (n, k, 2) with the per-frame homography (n, 3, 3)."""
    p = np.einsum("nij,nkj->nki", hom, np.stack([x, y, np.ones_like(x)], -1))
    return p[..., :2] / p[..., 2:3]


def main():
    m = from_tactical()
    hom = load_tactical()["hom"].astype(np.float64)
    n = len(m["bx"])
    assert len(hom) == n, (len(hom), n)
    px, py = m["px"].astype(np.float64), m["py"].astype(np.float64)
    pix = project(hom, px, py)
    pix[..., 1] -= LIFT
    c = np.full((n, 2), np.nan)
    ok = ~np.isnan(m["bx"])
    d = np.hypot(px - m["bx"][:, None], py - m["by"][:, None])
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argsort(d, axis=1)[:, :2]
    two = np.take_along_axis(pix, j[..., None], 1)
    good = ok & np.isfinite(np.take_along_axis(d, j, 1)).all(1)
    c[good] = two[good].mean(1)
    have = ~np.isnan(c[:, 0])
    idx = np.maximum.accumulate(np.where(have, np.arange(n), -1))
    idx[idx < 0] = np.argmax(have)
    c = c[idx]
    k = np.ones(25) / 25
    sm = np.stack([np.convolve(np.pad(c[:, i], 12, mode="edge"), k, mode="valid") for i in (0, 1)], 1)
    x0 = np.clip(np.round(sm[:, 0] - WIN_W / 2), 0, W - WIN_W).astype(np.int16)
    y0 = np.clip(np.round(sm[:, 1] - WIN_H / 2), 0, H - WIN_H).astype(np.int16)
    np.savez_compressed(OUT, x0=x0, y0=y0, pix=pix.astype(np.float16), slot_team=m["slot_team"], half=m["half"],
                        ball_seen=ok)
    print("frames", n, "ball seen {:.1%}".format(ok.mean()), "window centre on players {:.1%}".format(good.mean()),
          "x0 range", int(x0.min()), int(x0.max()), "y0 range", int(y0.min()), int(y0.max()), "->", OUT)


if __name__ == "__main__":
    main()
