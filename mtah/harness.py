"""Run a labeler over Matches and score it with the official metrics.

    from mtah.harness import evaluate
    table = evaluate(lambda m: control(m), matches, bootstrap=200)

One row per Match: spellF1 (mean over IoU 0.3-0.7, the ranking metric), macroF1, possession counts,
and the spellF1 bootstrap interval. Scoring is evaluation/possession_metrics.evaluate, unchanged.
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def evaluate(labeler, matches, bootstrap=200):
    rows = []
    for m in matches:
        r = pm.evaluate(m["labels"], labeler(m), bootstrap_n=bootstrap)
        rows.append({"name": m["name"], "spell_f1": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
                     "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"],
                     "boot_low": r["spell_bootstrap_low"], "boot_high": r["spell_bootstrap_high"]})
    return pd.DataFrame(rows)
