"""Aggregate possession diagnostics on labelled external matches, never a selection score."""

import numpy as np

from ..constants import H, A


def _spells(y, runs):
    out = []
    for lo, hi in runs:
        edges = np.r_[lo, np.flatnonzero(y[lo + 1:hi] != y[lo:hi - 1]) + lo + 1, hi]
        out.extend((int(s), int(e), int(y[s])) for s, e in zip(edges[:-1], edges[1:]) if y[s] in (H, A))
    return out


def _hits(spells, others, threshold):
    return [any(k == kk and min(e, b) > max(s, a)
                and (min(e, b) - max(s, a)) / (max(e, b) - min(s, a)) >= threshold
                for a, b, kk in others) for s, e, k in spells]


def diagnostics(gt, pred, runs, fps, iou=0.5):
    """Counts and best same-team IoU recall/precision by duration; no per-frame output.

    Both rates per live hour use GT live time, so adding predicted live frames cannot improve
    the denominator. IoU matching is diagnostic (not a replacement for the official metric).
    """
    truth, guessed = _spells(np.asarray(gt), runs), _spells(np.asarray(pred), runs)
    found, valid = _hits(truth, guessed, iou), _hits(guessed, truth, iou)
    live_hours = sum(e - s for s, e, _ in truth) / fps / 3600
    out = {"n_gt": len(truth), "n_pred": len(guessed), "iou": iou,
           "gt_per_live_hour": len(truth) / max(live_hours, 1e-9),
           "pred_per_gt_live_hour": len(guessed) / max(live_hours, 1e-9), "duration_bins": {}}
    for lo, hi, name in ((0, 1, "0-1s"), (1, 2, "1-2s"), (2, 4, "2-4s"), (4, np.inf, "4s+")):
        ti = [i for i, (s, e, _) in enumerate(truth) if lo <= (e - s) / fps < hi]
        pi = [i for i, (s, e, _) in enumerate(guessed) if lo <= (e - s) / fps < hi]
        out["duration_bins"][name] = {"n_gt": len(ti), "found": sum(found[i] for i in ti),
                                      "n_pred": len(pi), "matched_pred": sum(valid[i] for i in pi)}
    return out
