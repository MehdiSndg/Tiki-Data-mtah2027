"""Tiny synthetic exhaustive checks; no real probabilities or labels are inspected."""

import itertools

import numpy as np
import pytest

from mtah import config
from mtah.constants import H, A, D, X
from mtah.decoding.sequential import minimum_viterbi, decode_runs
from mtah.pipeline.adapt import blend_cfg, params


@pytest.mark.parametrize("classes", [2, 3])
def test_minimum_viterbi_matches_brute_force(classes):
    rng = np.random.default_rng(12)
    t = 6
    emissions = rng.normal(size=(t, classes))
    minimum = np.full(classes, 2, np.int64)
    transitions = -np.ones((classes, classes))
    bonus = rng.normal(size=t + 1)
    adjustments = rng.normal(size=(t, classes, classes)) * .1
    def objective(labels):
        changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        bounds = np.r_[0, changes, t]
        if any(b - a < 2 for a, b in zip(bounds[1:-2], bounds[2:-1])):
            return -np.inf
        return (emissions[np.arange(t), labels].sum()
                + sum(transitions[labels[i - 1], labels[i]] + bonus[i]
                      + adjustments[i, labels[i - 1], labels[i]] for i in changes))
    decoded = minimum_viterbi(emissions, minimum, transitions, bonus, adjustments)
    best = max(objective(np.array(y)) for y in itertools.product(range(classes), repeat=t))
    assert objective(decoded) == pytest.approx(best)


@pytest.mark.parametrize("mode", ["joint_minimum", "staged"])
def test_forced_dead_and_outside_runs_are_preserved(mode):
    cfg = config.load("ev_ha_swap_rf0")
    cfg = blend_cfg(cfg, params(cfg)[0])
    probs = np.tile([.99, .005, .005], (100, 1))
    labels = decode_runs(probs, [(10, 90)], cfg, mode, forced_dead=[(35, 45)])
    assert np.all(labels[:10] == X) and np.all(labels[90:] == X)
    assert np.all(labels[35:45] == D)
    assert np.isin(labels[10:90], [H, A, D]).all()


@pytest.mark.parametrize("mode", ["joint_minimum", "staged"])
def test_team_exchange_equivariance_and_inputs_unmodified(mode):
    cfg = config.load("ev_ha_swap_rf0")
    cfg = blend_cfg(cfg, params(cfg)[0])
    p = np.r_[np.tile([.96,.02,.02],(80,1)),np.tile([.02,.96,.02],(80,1)),np.tile([.02,.02,.96],(80,1))]
    before = p.copy()
    a = decode_runs(p, [(0,len(p))], cfg, mode)
    b = decode_runs(p[:,[A,H,D]], [(0,len(p))], cfg, mode)
    np.testing.assert_array_equal(np.where(a < D, 1-a, a), b)
    np.testing.assert_array_equal(p, before)
