"""The dead-ball classifier (analysis/m1_dead_classifier.py) on our match. GT-free checks first.

    uv run python analysis/m2_dead_on_our_match.py

Trained on ALL degraded external data (DFL 7 + Metrica 2), applied to our raw ATD. Then:
  - GT-free: dead share, number and median length of dead blocks (Metrica: ~37%, ~91 blocks per
    match, median 16-19 s; v0: 41%)
  - P1's periods are applied: frames of a half before its detected kickoff are dead (PRE2)
  - two dead-ball variants fixed before looking: classifier alone; classifier OR (broadcast cut
    away for >= 2 s while players are slow, speed_2s < 1.4 m/s)
  - control rule with each, on the public window: a correctness check only, with bootstrap
Writes evaluation/results/m2_clf.csv and m2_clf_bro.csv (gitignored) for the scoreboard.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import MIN_BLOCK, SEED, features, table, to_blocks  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.harness import evaluate  # noqa: E402
from mtah.labelers import control  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "evaluation", "results")


def blocks(dead):
    e = np.diff(np.r_[0, dead.astype(np.int8), 0])
    return (np.flatnonzero(e == -1) - np.flatnonzero(e == 1)) / FPS


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    train = [degrade(from_dfl(i), rates_for(i), seed=SEED) for i in ids] + \
            [degrade(from_metrica_sample(g), rates_for("metrica"), seed=SEED) for g in (1, 2)]
    X, y = table(train)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(X, y)

    m = from_tactical()
    g = m["half"] > 0
    F = features(m)
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(F[g])[:, 1]
    pre = np.isin(load_frames()["period"].values, NOT_PLAYED)   # before the kickoff, after the final whistle: known dead
    dead_clf = (to_blocks(p) | pre) & g
    b = np.load(os.path.join(ROOT, "cache", "broadcast_on_tactical.npz"))
    cut = b["b_present"] & ~b["b_valid"]
    cut2 = pd.Series(cut.astype(float)).rolling(MIN_BLOCK, center=True, min_periods=1).min().values > 0
    dead_bro = (dead_clf | (cut2 & (F["speed_2s"].values < 1.4) & (F["speed_2s"].values >= 0))) & g
    minutes = g.sum() / FPS / 60
    print("our match, {:.0f} graded minutes (GT-free):".format(minutes))
    for name, d in (("classifier", dead_clf), ("classifier + broadcast", dead_bro)):
        L = blocks(d)
        print("   {:<24} dead {:.0%}, {} blocks ({:.0f} per 90 min), median {:.1f} s".format(
            name, d[g].mean(), len(L), len(L) / minutes * 90, np.median(L)))
    print("   reference: Metrica sample ~37%, ~91 blocks per match, median 16-19 s; v0 41%")
    print("public window (correctness check only, bootstrap 1000):")
    for name, d, fn in (("classifier", dead_clf, "m2_clf"), ("classifier + broadcast", dead_bro, "m2_clf_bro")):
        codes = control(m, dead=d)
        codes = np.where(g, codes, pm.X).astype(np.int8)
        r = evaluate(lambda _: codes, [m], bootstrap=1000)
        print("   {:<24} spellF1 {:.3f} [{:.2f}-{:.2f}] macroF1 {:.3f} possessions {}/{}".format(
            name, r.spell_f1[0], r.boot_low[0], r.boot_high[0], r.macro_f1[0], r.n_pred[0], r.n_gt[0]))
        pd.DataFrame({"frame": np.arange(len(codes)), "label": np.array(pm.LABELS)[codes]}).to_csv(
            os.path.join(OUT, fn + ".csv"), index=False)


if __name__ == "__main__":
    main()
