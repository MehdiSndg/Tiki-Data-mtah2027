"""Region folds for the submission model (Gün 5, rapor 6 Q4; MASTER_PLAN Ek N.6).

The graded halves are cut into `chunks_per_half` chunks; every internal cut sits in the middle of
the ball-out-of-pitch run (ball > `oob_m` outside a line for >= `min_run` frames: a certain stoppage)
nearest to the equal-length cut. Chunks alternate between regions 0 .. n-1. A region-fold model k
never trains on region k's weak anchors (+- buffer) and predicts region k; stitching the folds
gives a submission in which no frame is predicted by a model that saw that frame's anchors
(Ek M.11 / rapor 6 F8: a model scores worse where it trained on the imperfect weak anchors).

The cuts depend on the tracking only -- never on labels/ (not committed), so the organisers'
re-run reproduces them. Labelled blocks need no special care: each of their frames is predicted by
the model that held its region out.
"""

from functools import lru_cache

import numpy as np

from ..constants import HALVES

DEFAULTS = dict(chunks_per_half=4, oob_m=0.35, min_run=10)


@lru_cache(maxsize=8)
def _chunks(chunks_per_half, oob_m, min_run):
    from ..io.atd import cached_barath
    d = cached_barath()
    L, W = d.pitch
    bx, by = d.ball[:, 0], d.ball[:, 1]
    with np.errstate(invalid="ignore"):
        out = np.nan_to_num(np.maximum.reduce([-bx, bx - L, -by, by - W]), nan=-1.0) > oob_m
    edges = np.flatnonzero(np.diff(np.r_[0, out.astype(np.int8), 0]))
    runs = [(s, e) for s, e in zip(edges[::2], edges[1::2]) if e - s >= min_run]
    mids = np.array([(s + e) // 2 for s, e in runs], np.int64)
    chunks = []
    for a, b in HALVES.values():
        cand = mids[(mids > a) & (mids < b)]
        cuts = []
        for i in range(1, chunks_per_half):
            nominal = a + (b - a) * i / chunks_per_half
            free = cand[~np.isin(cand, cuts)]
            if len(free) == 0:
                raise RuntimeError("no ball-out-of-pitch stoppage to cut a region at")
            cuts.append(int(free[np.argmin(np.abs(free - nominal))]))
        bounds = [a] + sorted(cuts) + [b]
        chunks += list(zip(bounds[:-1], bounds[1:]))
    return tuple((int(s), int(e)) for s, e in chunks)


def region_chunks(cfg=None):
    """[(start, stop, region)] over the graded halves for validation.region_folds folds."""
    v = (cfg or {}).get("validation") or {}
    rc = dict(DEFAULTS, **(v.get("regions") or {}))
    n = int(v.get("region_folds") or (v.get("region_cv") or [0, 2])[1])
    ch = _chunks(int(rc["chunks_per_half"]), float(rc["oob_m"]), int(rc["min_run"]))
    return [(s, e, i % n) for i, (s, e) in enumerate(ch)]


def region_of_frames(cfg, n_frames):
    """(n_frames,) int region per frame; frames outside the graded halves get region 0."""
    out = np.zeros(n_frames, np.int8)
    for s, e, r in region_chunks(cfg):
        out[s:e] = r
    return out
