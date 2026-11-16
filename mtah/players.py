"""Player arrays per team in metres, in an attack-direction frame, plus the camera's view box.

    t = team_arrays(m, half_of(code), attack_sign(meta))
    t["H_xy"][frame, k]      # (x, y) metres of the k-th home outfield slot, NaN when absent

The attack-direction frame puts the origin on the centre spot and has the HOME side attack +x in
both halves: when it really attacks -x the pitch is rotated 180 degrees (both axes reflected), so
left and right wings stay consistent too. Goalkeepers are their own arrays (slots in GK_SLOTS,
notes/VERI.md); slots that are practically never present are dropped.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import GK_SLOTS, PITCH  # noqa: E402

# Params
EMPTY_SLOT_SHARE = 0.01      # a slot present on fewer graded frames than this is dropped
IMG_W, IMG_H = 1920, 1080
GRID_STEP = 0.02             # normalised pitch grid projected into each frame for view_box
VIEW_CLIP = (-0.2, 1.2)
CHUNK = 4000                 # frames per vectorised block in view_box


def to_attack_frame(x, y, half, sign):
    """Normalised 0-1 pitch -> metres from the centre spot, home side attacking +x. NaN outside halves."""
    s = np.zeros(len(half), np.float64)
    for h, v in sign.items():
        s[np.asarray(half) == h] = v
    s[s == 0] = np.nan
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim > 1:
        s = s[:, None]
    return (x - 0.5) * PITCH[0] * s, (y - 0.5) * PITCH[1] * s


def team_arrays(m, half, sign):
    """Per-team (n, k, 2) outfield positions, (n, 2) goalkeeper positions and visible counts."""
    present = ~np.isnan(m["px"])
    share = present[m["graded"]].mean(0)
    X, Y = to_attack_frame(m["px"], m["py"], half, sign)
    out = {}
    for team, name in ((0, "H"), (1, "A")):
        gk = GK_SLOTS[team]
        cols = [s for s in np.flatnonzero(m["slot_team"] == team) if s != gk and share[s] >= EMPTY_SLOT_SHARE]
        out[name + "_xy"] = np.stack([X[:, cols], Y[:, cols]], axis=-1).astype(np.float32)
        out[name + "_gk"] = np.stack([X[:, gk], Y[:, gk]], axis=-1).astype(np.float32)
        out[name + "_n"] = (~np.isnan(X[:, cols])).sum(1).astype(np.int8)
    return out


def view_box(hom):
    """(n, 4) x_min, x_max, y_min, y_max of the pitch the camera shows, normalised, NaN if none.

    A grid over the pitch is projected into every frame; the box is the extent of the grid points
    that land inside the image, padded by half a grid step.
    """
    g = np.arange(0.0, 1.0 + 1e-9, GRID_STEP)
    gx, gy = [a.ravel() for a in np.meshgrid(g, g)]
    pts = np.stack([gx, gy, np.ones_like(gx)])                       # (3, p)
    out = np.full((len(hom), 4), np.nan)
    for a in range(0, len(hom), CHUNK):
        h = hom[a:a + CHUNK]                                          # (c, 3, 3)
        q = h @ pts                                                   # (c, 3, p)
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
        seen = (q[:, 2] > 0) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        any_seen = seen.any(1)
        for k, (coord, fn) in enumerate(((gx, np.min), (gx, np.max), (gy, np.min), (gy, np.max))):
            masked = np.where(seen, coord[None, :], np.nan)
            with np.errstate(all="ignore"):
                val = (np.nanmin if fn is np.min else np.nanmax)(masked[any_seen], axis=1)
            pad = -GRID_STEP / 2 if fn is np.min else GRID_STEP / 2
            out[a:a + CHUNK][any_seen, k] = np.clip(val + pad, *VIEW_CLIP)
    return out
