"""Paired comparison of two submissions on the same labelled blocks.

Both systems are scored against the SAME resampled set of ground-truth possessions in every
bootstrap draw (the resampling unit the official spellF1 bootstrap uses), so the result is the
probability that B really beats A rather than two overlapping intervals.
"""

import numpy as np

from .official import pm


def _flags(gt, pred, runs, gt_spells):
    filled = pm.fill_abstentions(np.asarray(pred, np.int8), runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), t) for t in pm.SPELL_IOUS])
    return flags, len(pred_spells)


def _f1(tp, n_pred, n_gt):
    fp = np.maximum(0, n_pred - tp)
    fn = n_gt - tp
    return np.where(tp > 0, 2 * tp / np.maximum(1, 2 * tp + fp + fn), 0.0)


def paired_spell_bootstrap(gt, pred_a, pred_b, n_draws=2000, seed=0):
    """{'p_b_better': P(spellF1_B > spellF1_A), 'mean_delta': E[B - A], 'delta': B - A observed}."""
    gt = np.asarray(gt, np.int8)
    runs = pm.valid_runs(gt)
    gt_spells = pm.possession_spells(gt, runs)
    n_gt = len(gt_spells)
    fa, na = _flags(gt, pred_a, runs, gt_spells)
    fb, nb = _flags(gt, pred_b, runs, gt_spells)
    rng = np.random.RandomState(seed)
    deltas = np.empty(n_draws)
    for i in range(n_draws):
        picks = rng.randint(0, n_gt, n_gt)
        sa = np.mean(_f1(fa[:, picks].sum(axis=1), na, n_gt))
        sb = np.mean(_f1(fb[:, picks].sum(axis=1), nb, n_gt))
        deltas[i] = sb - sa
    obs = np.mean(_f1(fb.sum(axis=1), nb, n_gt)) - np.mean(_f1(fa.sum(axis=1), na, n_gt))
    return {"p_b_better": float(np.mean(deltas > 0)), "p_tie": float(np.mean(deltas == 0)),
            "mean_delta": float(deltas.mean()), "delta": float(obs)}
