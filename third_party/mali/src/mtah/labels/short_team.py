"""Dense external-match targets for the optional short-possession team-loss experiment."""

import numpy as np

from ..constants import H, A


def frame_weights(labels, runs, fps, max_seconds=5.0, pad_frames=12, factor=2.0):
    """Weight short H/A spells and their neighbours, staying inside each played run.

    Durations are measured on full-resolution, already-aligned targets BEFORE chunking.
    The duration test is strict (< max_seconds). Dead/unknown frames always keep weight 1.
    Only use dense labels from other matches; sparse target-match anchors are not durations.
    """
    if fps <= 0 or max_seconds <= 0 or pad_frames < 0 or int(pad_frames) != pad_frames or factor < 1:
        raise ValueError("invalid short-possession weighting parameters")
    y = np.asarray(labels)
    out = np.ones(len(y), np.float32)
    radius = int(pad_frames)
    for a, b in runs:
        if not 0 <= a < b <= len(y):
            raise ValueError("invalid played run")
        bounds = np.r_[a, np.flatnonzero(y[a + 1:b] != y[a:b - 1]) + a + 1, b]
        marked = np.zeros(b - a, bool)
        for s, e in zip(bounds[:-1], bounds[1:]):
            if y[s] in (H, A) and e - s < fps * max_seconds:
                marked[max(a, s - radius) - a:min(b, e + radius) - a] = True
        marked &= np.isin(y[a:b], (H, A))
        out[a:b][marked] = factor
    return out


def step_weights(steps, labels, runs, fps, stride, **params):
    """Use exactly hybrid.step_view's centre-frame sampling, with no temporal rounding."""
    weights = frame_weights(labels, runs, fps, **params)
    return [weights[a + np.arange(len(x)) * stride + stride // 2].copy()
            for x, _, _, (a, _) in steps]
