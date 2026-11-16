"""The broadcast cutting away from the pitch as evidence for the dead ball, on our match. GT-free.

    uv run python analysis/p6n_dead_broadcast.py      # -> eda_out/reports/P6n_dead_broadcast.md

The dead-ball v2 classifier's probability (analysis/p4_common.dead_v2's model) gets naive-Bayes evidence
from the broadcast: the share of the centred 2 s window in which the broadcast shows no pitch
(cache/broadcast_on_tactical.npz: present and no valid homography). The likelihood ratio per bin of that
share is estimated on frames the classifier is sure of (p < 0.05 alive, p > 0.95 dead), outside the
public window; frames without broadcast get no evidence. Judged by an independent source, Smart
Tagging's set pieces (a restart follows a dead ball): the share of set pieces with a dead ball in the
5 s before them, against how many frames are called dead.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah.deadball import features_v2  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402
from p4_common import dead_classifier  # noqa: E402
from p4d_still_and_kinks import runs_of  # noqa: E402

BINS = (0, 0.01, 0.25, 0.5, 0.75, 1.01)
SCALES = (0.0, 0.5, 1.0)
OUT = os.path.join(ROOT, "eda_out", "reports", "P6n_dead_broadcast.md")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def main():
    m = from_tactical()
    fr = load_frames()
    g = m["half"] > 0
    clf = dead_classifier(featurize=features_v2)
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(features_v2(m)[g])[:, 1]
    b = np.load(os.path.join(ROOT, "cache", "broadcast_on_tactical.npz"))
    have, leave = b["b_present"], b["b_present"] & ~b["b_valid"]
    share = pd.Series(np.where(have, leave, np.nan).astype(float)).rolling(50, center=True, min_periods=10).mean().values
    outside = np.ones(len(g), bool)
    outside[11185:18685] = False
    sure_dead, sure_alive = g & outside & (p > 0.95), g & outside & (p < 0.05)
    E = np.zeros(len(g))
    lines = ["# P6n: yayının sahadan ayrılması ölü top kanıtı olarak (GT'siz)", "", "`analysis/p6n_dead_broadcast.py`.", "",
             "| 2 s'de ayrılma payı | emin ölü | emin oyun | log olabilirlik oranı |", "|---|---|---|---|"]
    for a, z in zip(BINS[:-1], BINS[1:]):
        s = ~np.isnan(share) & (share >= a) & (share < z)
        nd, na = (s & sure_dead).sum(), (s & sure_alive).sum()
        lr = np.log(((nd + 1) / (sure_dead.sum() + 1)) / ((na + 1) / (sure_alive.sum() + 1)))
        E[s] = lr
        lines.append("| {:.2f}–{:.2f} | {} | {} | {:+.2f} |".format(a, min(z, 1), nd, na, lr))
    starts = [a for a, _ in runs_of(fr["setpiece"].values.astype(bool)) if outside[a] and g[a]]
    notplayed = np.isin(fr["period"].values, NOT_PLAYED)
    lines += ["", "| ölçek | ölü kare payı | duran topların önünde ölü top (5 s) |", "|---|---|---|"]
    for sc in SCALES:
        q = 1 / (1 + np.exp(-(logit(p) + sc * E)))
        dead = (to_blocks(np.where(g, q, 0)) | notplayed) & g
        cov = np.mean([dead[max(0, a - 125):a + 1].any() for a in starts])
        lines.append("| {:g} | {:.1%} | {:.1%} ({}) |".format(sc, dead[g].mean(), cov, len(starts)))
        print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
