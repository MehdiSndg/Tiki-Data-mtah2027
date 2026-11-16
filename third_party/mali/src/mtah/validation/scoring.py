"""Scoring with the official metric on labelled blocks, plus readable reports."""

import numpy as np

from ..constants import X
from .official import pm
from .splits import labelled_blocks, merged_gt


def evaluate(pred_codes, gt_codes, bootstrap=0):
    return pm.evaluate(np.asarray(gt_codes, np.int8), np.asarray(pred_codes, np.int8),
                       bootstrap_n=bootstrap)


def score_blocks(pred_codes, blocks=None, bootstrap=0):
    """{'ALL': result, '<block>': result, ...} on the labelled validation blocks."""
    blocks = blocks if blocks is not None else labelled_blocks()
    out = {"ALL": evaluate(pred_codes, merged_gt(blocks), bootstrap)}
    if len(blocks) > 1:
        for block, codes in blocks:
            out[block.name] = evaluate(pred_codes, merged_gt([(block, codes)]), bootstrap)
    return out


def summary_line(name, r):
    d = r["per_class"]["D"]
    line = (f"{name:<22} spellF1 {r['spell_f1_mean']:.4f}  macroF1 {r['macro_f1']:.4f}  "
            f"poss {r['n_pred_spells']:>4}/{r['n_gt_spells']:<4} dur {r['mean_spell_duration_s']:5.1f}s"
            f"  D P/R {d['precision']:.2f}/{d['recall']:.2f}  team-acc {r['team_acc_both_alive']:.3f}"
            f"  transF1@25 {r['transitions'][25]['f1']:.3f}")
    if r.get("spell_bootstrap_low") is not None:
        line += f"  [boot {r['spell_bootstrap_low']:.3f}..{r['spell_bootstrap_high']:.3f}]"
    return line


def report(results, title):
    lines = [title]
    for name, r in results.items():
        lines.append("  " + summary_line(name, r))
    return "\n".join(lines)


def assert_no_graded_x(codes, runs):
    for a, b in runs:
        if np.any(np.asarray(codes)[a:b] == X):
            raise AssertionError("X inside a graded run")
