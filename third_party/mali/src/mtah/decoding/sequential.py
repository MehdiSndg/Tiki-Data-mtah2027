"""CPU-only staged decoding diagnostic; emissions and trained models are frozen.

The joint minimum-dwell control removes soft duration priors too, so any staged gain
must also be compared with that control. No parameter grid and no learned dead model.
"""

import math

import numpy as np
from numba import njit

from ..constants import H, A, D, X
from . import hsmm


@njit(cache=True)
def minimum_viterbi(emissions, minimum, transitions, bonus, adjustments):
    """Minimum dwell with unrestricted first/last segments, deterministic ties."""
    steps, classes = emissions.shape
    offsets = np.zeros(classes + 1, np.int64)
    for k in range(classes):
        offsets[k + 1] = offsets[k] + minimum[k]
    states = offsets[-1]
    owners = np.empty(states, np.int64)
    for k in range(classes):
        owners[offsets[k]:offsets[k + 1]] = k
    previous = np.full(states, -1e300)
    back = np.full((steps, states), -1, np.int32)
    for k in range(classes):
        previous[offsets[k + 1] - 1] = emissions[0, k]
    for t in range(1, steps):
        current = np.full(states, -1e300)
        for k in range(classes):
            start, mature = offsets[k], offsets[k + 1] - 1
            for dest in range(start + 1, mature + 1):
                current[dest] = previous[dest - 1]
                back[t, dest] = dest - 1
            if previous[mature] > current[mature]:
                current[mature] = previous[mature]
                back[t, mature] = mature
            for other in range(classes):
                if other == k:
                    continue
                src = offsets[other + 1] - 1
                value = previous[src] + transitions[other, k] + bonus[t] + adjustments[t, other, k]
                if value > current[start]:
                    current[start] = value
                    back[t, start] = src
            for dest in range(start, mature + 1):
                current[dest] += emissions[t, k]
        previous = current
    q = int(np.argmax(previous))
    labels = np.empty(steps, np.int8)
    for t in range(steps - 1, -1, -1):
        labels[t] = owners[q]
        if t:
            q = back[t, q]
    return labels


def _decode(lp, cfg, minimum, transitions, boundary, local):
    if len(lp) == 0:
        return np.zeros(0, np.int8)
    stride = int(cfg["decoder"]["stride"])
    steps = math.ceil(len(lp) / stride)
    padding = steps * stride - len(lp)
    padded = np.vstack([lp, np.repeat(lp[-1:], padding, axis=0)]) if padding else lp
    step_lp = padded.reshape(steps, stride, lp.shape[1]).sum(1)
    bonus = hsmm.boundary_bonus(boundary, stride, steps, cfg["decoder"].get("boundary_gamma", 0.))
    adjustments = np.zeros((steps, lp.shape[1], lp.shape[1]))
    for (a, b), values in local.items():
        adjustments[:, a, b] = hsmm._step_mean(values, stride, steps)
    mins = np.array([max(1, math.ceil(m / stride)) for m in minimum], np.int64)
    labels = minimum_viterbi(step_lp, mins, np.asarray(transitions, np.float64), bonus, adjustments)
    labels = np.repeat(labels, stride)[:len(lp)].astype(np.int8)
    return hsmm._refine(labels, lp, int(cfg["decoder"]["refine_frames"]))


def decode_run(probs, cfg, mode, boundary=None, regime=None, forced=None):
    if mode not in ("joint_minimum", "staged"):
        raise ValueError("Unknown sequential diagnostic mode")
    dc = cfg["decoder"]
    lp = hsmm.emission_logprob(probs, cfg)
    minimum_team, minimum_dead = dc["duration"]["possession"]["min"], dc["duration"]["dead"]["min"]
    regime = regime or {}
    stop, restart, turnover = dc["penalty_stoppage"], dc["penalty_restart"], dc["penalty_turnover"]
    if forced is not None and forced.any():
        lp[forced, :2], lp[forced, D] = -1e4, 0.
    if mode == "joint_minimum":
        local = {(H, A): regime.get("turnover_adj"), (A, H): regime.get("turnover_adj"),
                 (H, D): regime.get("stoppage_adj"), (A, D): regime.get("stoppage_adj"),
                 (D, H): regime.get("restart_adj"), (D, A): regime.get("restart_adj")}
        labels = _decode(lp, cfg, [minimum_team, minimum_team, minimum_dead],
                         [[0, -turnover, -stop], [-turnover, 0, -stop], [-restart, -restart, 0]], boundary, local)
    else:
        p = np.clip(np.asarray(probs, np.float64), dc["prob_floor"], 1.)
        dead_lp = np.c_[np.log(np.clip(p[:, H] + p[:, A], dc["prob_floor"], 1.)), np.log(p[:, D])]
        dead_lp *= np.array([1., dc["emission_weights"][D]]) * dc.get("emission_scale", 1.)
        dead_lp = np.round(dead_lp, 6)
        if forced is not None and forced.any():
            dead_lp[forced, 0], dead_lp[forced, 1] = -1e4, 0.
        dead = _decode(dead_lp, cfg, [minimum_team, minimum_dead], [[0, -stop], [-restart, 0]], boundary,
                       {(0, 1): regime.get("stoppage_adj"), (1, 0): regime.get("restart_adj")}).astype(bool)
        if forced is not None:
            dead |= forced
        labels = np.full(len(lp), D, np.int8)
        edges = np.flatnonzero(np.diff(np.r_[False, ~dead, False]))
        for a, b in zip(edges[::2], edges[1::2]):
            adj = regime.get("turnover_adj")
            labels[a:b] = _decode(lp[a:b, :2], cfg, [minimum_team] * 2,
                                  [[0, -turnover], [-turnover, 0]],
                                  None if boundary is None else boundary[a:b],
                                  {(0, 1): None if adj is None else adj[a:b],
                                   (1, 0): None if adj is None else adj[a:b]})
    if forced is not None:
        labels[forced] = D
    return labels


def decode_runs(probs, runs, cfg, mode, boundary=None, regime=None, forced_dead=()):
    labels = np.full(len(probs), X, np.int8)
    for a, b in runs:
        forced = np.zeros(b - a, bool)
        for lo, hi in forced_dead:
            start, end = max(lo, a), min(hi, b)
            if start < end:
                forced[start - a:end - a] = True
        local = {k: None if value is None else value[a:b] for k, value in (regime or {}).items()}
        labels[a:b] = decode_run(probs[a:b], cfg, mode, None if boundary is None else boundary[a:b], local, forced)
    return labels
