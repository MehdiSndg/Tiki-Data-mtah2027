import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.hsmm import _run, decode_hsmm  # noqa: E402


def brute(e, lp, cost):
    n, best, arg = len(e), -np.inf, None
    for states in itertools.product((0, 1), repeat=n):
        segs = pm.segments(np.array(states, np.int8), [(0, n)])
        v = sum(e[a:b, k].sum() + lp[b - a] for a, b, k in segs) - sum(cost[a] for a, _, _ in segs[1:])
        if v > best:
            best, arg = v, states
    return np.array(arg)


def test_run_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = 7
        e = rng.normal(0, 1, (n, 2))
        lp = np.r_[-1e9, rng.normal(-2, 1, n)]
        cost = rng.uniform(0, 1, n)
        assert _run(e, lp, cost, dmax=n).tolist() == brute(e, lp, cost).tolist()


def test_decode_respects_stoppages_and_strong_evidence():
    p = np.r_[np.full(50, 0.95), np.full(10, 0.5), np.full(50, 0.05)]
    alive = np.ones(110, bool)
    alive[50:60] = False
    lp = np.log(np.full(200, 1 / 200))
    c = decode_hsmm(p, alive, lp)
    assert (c[:50] == pm.H).all() and (c[50:60] == pm.D).all() and (c[60:] == pm.A).all()
