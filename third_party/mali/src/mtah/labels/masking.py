"""Self-supervised ball masking: hide the ball in realistic blocks so models learn the gaps.

Block lengths are drawn from the source's own empirical gap-length distribution inside played
periods (BAR-ATH: median ~10.7 s, long tail to minutes), so masked frames look like real gaps to
every feature (gap length, edge anchors, rolling context).

Two modes:
  mask_ball           uniform: blocks start anywhere in visible play (BAR-ATH copies: the true
                      state is unknown there, so masking must not depend on it).
  mask_ball_by_state  state-aware, for sources WITH true labels (DFL): blocks start inside dead
                      segments until `dead_fraction` of the visible dead frames is hidden, then
                      anywhere in live play up to `live_fraction`. On BAR-ATH the ball is missing
                      on ~58% of predicted-dead and ~32% of predicted-live graded frames (M0
                      ensemble, no GT used), while clean DFL always shows the ball, also in dead
                      play. Without this a model trained on DFL learns "visible ball = dead
                      ball is placed" cues that ATD never shows. Blocks are not clipped at the
                      state boundary (ATD gaps also run across restarts), so gap edges do not
                      reveal the label.
"""

import numpy as np


def gap_lengths(data):
    vis = ~np.isnan(data.ball[:, 0])
    playing = data.period > 0
    gaps = (~vis) & playing
    edges = np.diff(np.concatenate(([0], gaps.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = stops - starts
    return lengths[lengths > 0]


def _fill(masked, eligible, starts, goal_mask, goal, rng, lengths, n):
    """Add blocks starting in `starts` until goal_mask & masked has `goal` frames."""
    if len(starts) == 0:
        return masked
    while int((masked & goal_mask).sum()) < goal:
        start = int(rng.choice(starts))
        stop = min(n, start + int(rng.choice(lengths)))
        masked[start:stop] |= eligible[start:stop]
    return masked


def mask_ball(data, rng, target_fraction, lengths=None):
    """(masked copy, newly_masked bool mask). Masks ~target_fraction of visible played frames."""
    lengths = gap_lengths(data) if lengths is None else lengths
    if len(lengths) == 0:
        lengths = np.array([250])
    out = data.copy()
    eligible = ~np.isnan(data.ball[:, 0]) & (data.period > 0)
    candidates = np.flatnonzero(eligible)
    masked = _fill(np.zeros(data.n_frames, bool), eligible, candidates, eligible,
                   int(target_fraction * len(candidates)), rng, lengths, data.n_frames)
    out.ball[masked] = np.nan
    return out, masked


def mask_ball_by_state(data, rng, labels, dead_fraction, live_fraction, lengths, dead_code=2):
    """(masked copy, newly_masked). Hides ~dead_fraction of visible dead frames and at least
    ~live_fraction of visible live frames (blocks that start in dead play may spill into live)."""
    lengths = np.asarray(lengths) if len(lengths) else np.array([250])
    out = data.copy()
    eligible = ~np.isnan(data.ball[:, 0]) & (data.period > 0)
    dead = eligible & (np.asarray(labels) == dead_code)
    live = eligible & ~dead
    masked = np.zeros(data.n_frames, bool)
    masked = _fill(masked, eligible, np.flatnonzero(dead), dead, int(dead_fraction * dead.sum()),
                   rng, lengths, data.n_frames)
    masked = _fill(masked, eligible, np.flatnonzero(live), live, int(live_fraction * live.sum()),
                   rng, lengths, data.n_frames)
    out.ball[masked] = np.nan
    return out, masked
