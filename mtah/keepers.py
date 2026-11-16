"""Is a goalkeeper slot backed by a person in the image? (P1.1, decided 22 Oct)

    from mtah.keepers import keeper_status, load_persons
    persons, covered = load_persons()
    status = keeper_status(m["px"][:, 23], m["py"][:, 23], m["hom"], persons, covered, PITCH)

The ATD keeps a keeper slot filled ~93% of the time; about half of that the keeper is off-screen and
the slot holds a stale position, and while on screen the slot has no person under it on 20% (home)
and 10% (away) of frames (notes/BULGULAR.md, notes/BULGULAR.md). Per frame:

    SEEN       on screen, a YOLO person's foot within RADIUS m
    GHOST      on screen, nobody within RADIUS m: the slot is not where the keeper is
    OFFSCREEN  outside the image: the ATD's last known position, unverifiable
    ABSENT     the slot is empty
    UNKNOWN    on screen, but the frame was not run through YOLO

Persons come from the pretrained YOLO pass (analysis/p3_yolo_pass.py, cache/yolo/); the
recovery of the keeper's real position from them is P3's job.
"""

import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Params
SEEN, GHOST, OFFSCREEN, ABSENT, UNKNOWN = 0, 1, 2, 3, 4
YOLO_DIR = os.path.join(ROOT, "cache", "yolo")
PERSON_CONF = 0.3           # the pass keeps persons from 0.30
RADIUS = 2.0                # metres; the morning check used 2 m (outfield slots fail it on 0.4%)
PREFILTER_PX = 300          # a person farther than this in the image is farther than RADIUS anywhere
IMG_W, IMG_H, EDGE = 1920, 1080, 40


def load_persons(yolo_dir=YOLO_DIR):
    """(persons: DataFrame frame, u, v = foot point in pixels; covered: frames run through YOLO)."""
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(yolo_dir, "chunk_*.npz")))]
    if not parts:
        return pd.DataFrame({"frame": [], "u": [], "v": []}), np.array([], int)
    cols = {c: np.concatenate([p[c] for p in parts]) for c in ("frame", "cls", "conf", "x", "y", "h")}
    keep = (cols["cls"] == 0) & (cols["conf"] >= PERSON_CONF)
    persons = pd.DataFrame({"frame": cols["frame"][keep].astype(int), "u": cols["x"][keep],
                            "v": cols["y"][keep] + cols["h"][keep] / 2})
    covered = np.unique(np.concatenate([np.arange(p["frame"].min(), p["frame"].max() + 1) for p in parts if len(p["frame"])]))
    return persons, covered


def keeper_status(x, y, hom, persons, covered, pitch):
    L, W = pitch
    n = len(x)
    d = hom[:, 2, 0] * x + hom[:, 2, 1] * y + hom[:, 2, 2]
    u = (hom[:, 0, 0] * x + hom[:, 0, 1] * y + hom[:, 0, 2]) / d
    v = (hom[:, 1, 0] * x + hom[:, 1, 1] * y + hom[:, 1, 2]) / d
    absent = np.isnan(x) | np.isnan(y)
    with np.errstate(invalid="ignore"):
        on_screen = ~absent & (d > 0) & (u > EDGE) & (u < IMG_W - EDGE) & (v > EDGE) & (v < IMG_H - EDGE)
    is_covered = np.zeros(n, bool)
    is_covered[covered[covered < n]] = True

    nearest = np.full(n, np.inf)
    f = persons["frame"].values
    pu, pv = persons["u"].values, persons["v"].values
    ok = (f < n)
    f, pu, pv = f[ok], pu[ok], pv[ok]
    cand = on_screen[f] & (np.hypot(pu - u[f], pv - v[f]) < PREFILTER_PX)
    f, pu, pv = f[cand], pu[cand], pv[cand]
    if len(f):
        Hi = np.linalg.inv(hom[f])
        q = np.einsum("nij,nj->ni", Hi, np.c_[pu, pv, np.ones(len(f))])
        X, Y = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
        np.minimum.at(nearest, f, np.hypot((X - x[f]) * L, (Y - y[f]) * W))

    status = np.full(n, UNKNOWN, np.int8)
    status[on_screen & is_covered] = np.where(nearest[on_screen & is_covered] <= RADIUS, SEEN, GHOST)
    status[~absent & ~on_screen] = OFFSCREEN
    status[absent] = ABSENT
    return status
