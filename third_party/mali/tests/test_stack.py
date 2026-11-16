"""Autonomous loop: calibration / stacking helpers and the Report-3 post-processor."""

import numpy as np

from mtah.constants import A, D, H, X
from mtah.decoding import post
from mtah.pipeline import stack


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def test_vector_scaling_undoes_a_known_temperature():
    rng = np.random.default_rng(0)
    n = 20_000
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 2.0
    true = _softmax(logits)
    y = np.array([rng.choice(3, p=p) for p in true])             # labels drawn from the truth
    over = _softmax(logits * 3.0)                                   # over-confident model (T = 1/3)
    th = stack.vector_scale_fit(over, y)
    assert 2.4 < np.exp(th[0]) < 3.6                                # recovers T ~ 3
    nll = lambda p: -np.mean(np.log(p[np.arange(n), y]))           # noqa: E731
    assert nll(stack.vector_scale_apply(over, th)) < nll(over)


def test_em_prior_shift_moves_towards_the_test_prior():
    rng = np.random.default_rng(1)
    p = _softmax(rng.normal(0, 1, (5_000, 3)) + np.array([0, 0, 1.5]))
    adj, prior = stack.em_prior_shift(p, np.array([1 / 3, 1 / 3, 1 / 3]))
    assert np.allclose(adj.sum(1), 1) and prior[2] > p[:, 2].mean() - 1e-9


def test_stacker_fit_predict_shapes_and_nesting():
    rng = np.random.default_rng(2)
    items = {}
    for m in ("a", "b", "c"):
        y = np.repeat(rng.integers(0, 3, 60), 50).astype(np.int8)
        y[:100] = X
        pb = _softmax(np.eye(3)[np.clip(y, 0, 2)] * 2 + rng.normal(0, 1, (len(y), 3)))
        pn = _softmax(np.eye(3)[np.clip(y, 0, 2)] * 3 + rng.normal(0, 1, (len(y), 3)))
        items[m] = (pb, pn, np.ones(len(y), bool), y)
    for mode in ("pool", "meta"):
        out = stack.nested_dfl(items, lambda: stack.Stacker(mode=mode, em=True))
        assert set(out) == {"a", "b", "c"}
        for m, p in out.items():
            assert p.shape == (len(items[m][3]), 3) and np.allclose(p.sum(1), 1, atol=1e-5)


def test_report_postprocess_closes_gap_and_snaps_turnover():
    n = 300
    lab = np.array([H] * 100 + [D] * 40 + [H] * 60 + [A] * 100, np.int8)
    ball = np.full((n, 2), 30.0, np.float32)
    acc = np.zeros(n)
    acc[192] = 12.0                                                 # kick 8 frames before the turnover
    near_h = np.full(n, 5.0)
    near_a = np.full(n, 5.0)
    near_a[185:200] = 1.0                                           # away player at the ball
    probs = np.full((n, 3), 1 / 3)
    out = post.report_postprocess(lab, [(0, n)], probs, ball, (105.0, 68.0), acc, near_h, near_a)
    assert np.all(out[100:140] == H)                                # false dead gap closed
    assert out[191] == H and out[192] == A                          # turnover snapped to the kick
