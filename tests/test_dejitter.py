"""mtah.dejitter: DFL's raw player tracks jitter frame to frame (per-frame acceleration median 6.3 m/s² vs our ATD's 0.8,
13 Nov scan) while 5-frame speeds agree; a centred moving average within each unbroken track piece removes it."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from test_degrade import toy_match  # noqa: E402


def noisy(n=2000, sd=0.3):
    m = toy_match(n)
    rng = np.random.default_rng(0)
    m["px"] = (m["px"] + rng.normal(0, sd / 105, m["px"].shape)).astype(np.float32)
    m["py"] = (m["py"] + rng.normal(0, sd / 68, m["py"].shape)).astype(np.float32)
    return m


def test_acc_median_of_a_constant_velocity_track_is_zero():
    from mtah.dejitter import player_acc_median
    m = toy_match(500)
    t = np.arange(500)
    m["px"][:] = (0.1 + 0.0004 * t)[:, None]
    m["py"][:] = 0.5
    assert player_acc_median(m) < 1e-3


def test_dejitter_lowers_the_acceleration_of_a_jittery_track():
    from mtah.dejitter import dejitter, player_acc_median
    m = noisy()
    assert player_acc_median(dejitter(m, 9)) < 0.3 * player_acc_median(m)


def test_dejitter_keeps_a_constant_velocity_track_inside_its_piece():
    from mtah.dejitter import dejitter
    m = toy_match(500)
    t = np.arange(500)
    m["px"][:] = (0.1 + 0.0004 * t)[:, None]
    out = dejitter(m, 9)
    assert np.allclose(out["px"][10:-10], m["px"][10:-10], atol=1e-6)


def test_dejitter_keeps_gaps_and_does_not_smooth_across_them():
    from mtah.dejitter import dejitter
    m = toy_match(600)
    m["px"][200:300, 4] = np.nan
    m["px"][:200, 4], m["px"][300:, 4] = 0.2, 0.8                 # a jump across the gap
    out = dejitter(m, 9)
    assert np.isnan(out["px"][200:300, 4]).all()
    assert np.allclose(out["px"][:200, 4], 0.2) and np.allclose(out["px"][300:, 4], 0.8)


def test_dejitter_does_not_smooth_across_the_half_break():
    from mtah.dejitter import dejitter
    m = toy_match(600)
    m["half"][300:] = 2
    m["px"][:300, 4], m["px"][300:, 4] = 0.2, 0.8
    out = dejitter(m, 9)
    assert np.allclose(out["px"][:300, 4], 0.2) and np.allclose(out["px"][300:, 4], 0.8)


def test_dejitter_leaves_the_ball_and_labels_alone():
    from mtah.dejitter import dejitter
    m = noisy()
    out = dejitter(m, 9)
    assert np.array_equal(out["bx"], m["bx"]) and np.array_equal(out["labels"], m["labels"])
