"""decoding.mbr: sampler agrees with Viterbi at low temperature; selection obeys the spell rules."""
import numpy as np

from mtah import config as config_mod
from mtah.decoding import hsmm, mbr


def _tables(cfg, lp):
    stride, dur, edge, dmin, trans = hsmm._tables(cfg)
    S = len(lp) // stride
    step_lp = lp[:S * stride].reshape(S, stride, 3).sum(axis=1)
    z = np.zeros(S + 1)
    return step_lp, dur, edge, dmin, trans, z, np.zeros((S + 1, 3)), np.zeros(S + 1, np.int64), np.full(dur.shape[1], hsmm.NEG)


def _toy(seed=0, n=6000):
    rng = np.random.default_rng(seed)
    true = np.repeat(rng.integers(0, 3, n // 300), 300)
    p = np.full((n, 3), 0.15)
    p[np.arange(n), true] = 0.7
    p += rng.uniform(0, 0.1, p.shape)
    return p / p.sum(1, keepdims=True)


def test_cold_samples_equal_viterbi():
    cfg = config_mod.load("m0")
    lp = hsmm.emission_logprob(_toy(), cfg)
    t = _tables(cfg, lp)
    vit = hsmm._viterbi(*t)
    s = mbr.sample_paths(*t, 0.02, 5, 1)
    assert (s == vit[None]).mean() > 0.999


def test_selection_rules_and_determinism():
    cfg = config_mod.load("m0")
    lp = hsmm.emission_logprob(_toy(3), cfg)
    t = _tables(cfg, lp)
    a = mbr.decode(*t, dict(samples=60, temperature=20.0, seed=5))
    b = mbr.decode(*t, dict(samples=60, temperature=20.0, seed=5))
    assert (a == b).all() and set(np.unique(a)) <= {0, 1, 2}
    sp = mbr.spells(a)
    assert len(sp) > 0 and (sp[1:, 0] >= sp[:-1, 1]).all()


def test_select_prefers_high_credit_and_separates_same_team():
    cand = np.array([[0, 10, 0], [10, 20, 0], [0, 20, 0], [20, 30, 1]], np.int64)
    cand = cand[np.lexsort((cand[:, 0], cand[:, 1]))]
    val = {(0, 10, 0): 0.6, (10, 20, 0): 0.6, (0, 20, 0): 0.9, (20, 30, 1): 0.5}
    v = np.array([val[tuple(c)] for c in cand])
    got = {tuple(cand[i]) for i in mbr._select(cand, v, 30)}
    assert got == {(0, 20, 0), (20, 30, 1)}       # the two touching H spells cannot both be chosen
