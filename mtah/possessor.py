"""Possessor-crop evidence for the team call (p6d's classifier on every person box). (step 3c)

    from mtah.possessor import crop_evidence, calibrate, evidence_logodds, SCALE
    t, p = crop_evidence(m)                 # per frame: team 0/1 of the most likely possessor box, its probability
    lo = calibrate(m, t, p, frames)         # GT-free calibration -> log-odds per probability bin
    E = evidence_logodds(t, p, lo)          # + favours home; enters the Viterbi as SCALE * E

m needs "hom" (pitch -> pixel homography per frame) besides the Match fields. See analysis/p6f_fuse_crops.py
for how SCALE was chosen without ground truth (ball-dropout test on our match).
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from mtah.data import PITCH  # noqa: E402
from mtah.kits import slot_labels  # noqa: E402

PROB = os.path.join(ROOT, "cache", "possessor_prob.npz")
TOP_MIN = 0.5
BINS = (0.5, 0.7, 0.9, 0.97, 1.01)
SCALE = 0.2                 # P6f ball-dropout test: best of 0, 0.05, 0.1, 0.2, 0.4, 0.8


def crop_evidence(m):
    """(team, prob) per frame: team 0/1 of the most likely possessor box, -1 when none."""
    from p3a_yolo_ball import to_pitch
    d = np.load(PROB)
    b = pd.DataFrame({k: d[k] for k in ("frame", "x", "y", "w", "h", "prob")})
    b = b[b.prob >= TOP_MIN]
    b = b.sort_values("prob").groupby("frame").tail(1)
    foot = pd.DataFrame({"frame": b.frame.values, "x": b.x.values, "y": b.y.values + b.h.values / 2})
    X, Y = to_pitch(foot, m["hom"])
    f = b.frame.values
    lab = slot_labels(np.c_[X, Y], m["px"][f] * PITCH[0], m["py"][f] * PITCH[1], m["slot_team"], m["gk"])
    team = np.where(lab >= 2, lab - 2, lab)
    n = len(m["bx"])
    t, p = np.full(n, -1), np.zeros(n)
    t[f], p[f] = team, b.prob.values
    return t, p


def calibrate(m, t, p, val):
    """log-odds per BINS bin that the evidence's team is the team of the player clearly nearest the slow ATD ball."""
    L, W = PITCH
    bx, by = m["bx"] * L, m["by"] * W
    sp = np.full(len(bx), np.nan)
    sp[5:] = np.hypot(bx[5:] - bx[:-5], by[5:] - by[:-5]) * 5
    d = np.hypot(m["px"] * L - bx[:, None], m["py"] * W - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    k = np.argmin(d, axis=1)
    near = d[np.arange(len(k)), k]
    second = np.sort(d, axis=1)[:, 1]
    ref = np.where((near <= 1.0) & (second >= 2.0) & (np.nan_to_num(sp, nan=99) < 6), m["slot_team"][k], -1)
    sel = val[(ref[val] >= 0) & (t[val] >= 0)]
    lo = []
    for a, z in zip(BINS[:-1], BINS[1:]):
        s = sel[(p[sel] >= a) & (p[sel] < z)]
        acc = (np.sum(t[s] == ref[s]) + 1) / (len(s) + 2)
        lo.append((a, z, len(s), acc, np.log(acc / (1 - acc))))
    return lo


def evidence_logodds(t, p, lo):
    e = np.zeros(len(t))
    for a, z, _, _, w in lo:
        s = (t >= 0) & (p >= a) & (p < z)
        e[s] = np.where(t[s] == 0, w, -w)          # + for home
    return e
