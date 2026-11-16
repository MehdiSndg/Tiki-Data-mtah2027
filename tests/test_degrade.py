import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import (DEFAULT_RATES, ball_drift, degrade, double_count, drop_gk_hold,  # noqa: E402
                          drop_stationary_ball, fragment_players, freeze_keepers, lose_ball, rates_for, view_window)


def toy_match(n=3000, seed=0):
    """22 players on random walks, keepers near their goals, a ball moving at ~8 m/s along x."""
    rng = np.random.RandomState(seed)
    px = np.clip(rng.uniform(0.2, 0.8, 22) + np.cumsum(rng.normal(0, 0.0005, (n, 22)), axis=0), 0, 1)
    py = np.clip(rng.uniform(0.1, 0.9, 22) + np.cumsum(rng.normal(0, 0.0005, (n, 22)), axis=0), 0, 1)
    px[:, 0], px[:, 11] = 0.03, 0.97
    t = np.arange(n)
    bx = 0.5 + 0.4 * np.sin(t / 400.0)
    return {"name": "toy", "px": px.astype(np.float32), "py": py.astype(np.float32),
            "slot_team": np.r_[np.zeros(11), np.ones(11)].astype(np.int8), "gk": {0: 0, 1: 11},
            "bx": bx.astype(np.float32), "by": np.full(n, 0.5, np.float32),
            "half": np.ones(n, np.int8), "labels": np.full(n, pm.H, np.int8),
            "sign": {1: 1, 2: -1}, "pitch": (105.0, 68.0)}


def rates(**kw):
    r = dict(DEFAULT_RATES)
    r.update(kw)
    return r


def test_view_window_keeps_the_ball_and_drops_far_players():
    m = toy_match()
    m["px"][:, 5] = np.where(m["bx"] < 0.5, 0.99, 0.01)        # always at the far end from the ball
    out = view_window(m, np.random.RandomState(0), rates(view_width=(0.4, 0.4, 0.4)))
    assert not np.isnan(out["bx"]).any()
    assert np.isnan(out["px"][:, 5]).all()
    assert out["px"].shape == m["px"].shape


def test_stationary_ball_removed():
    m = toy_match(n=200)
    m["bx"][100:120] = m["bx"][100]                           # 20 frames still
    m["bx"][150:155] = m["bx"][150]                           # 5 frames still: too short
    out = drop_stationary_ball(m, np.random.RandomState(0), rates(dead_ball_drop=0.0, still_frames=10))
    assert np.isnan(out["bx"][101:119]).all()
    assert not np.isnan(out["bx"][:95]).any() and not np.isnan(out["bx"][150:155]).any()


def test_dead_ball_dropped_by_runs():
    m = toy_match(n=400)
    m["labels"][100:200] = pm.D
    out = drop_stationary_ball(m, np.random.RandomState(0), rates(dead_ball_drop=1.0))
    assert np.isnan(out["bx"][100:200]).all() and not np.isnan(out["bx"][:100]).any()


def test_gk_hold_removed():
    m = toy_match(n=300)
    m["bx"][50:80], m["by"][50:80] = m["px"][50:80, 0] + 0.003, m["py"][50:80, 0]   # 0.3 m, 1.2 s
    m["bx"][200:205], m["by"][200:205] = m["px"][200:205, 0], m["py"][200:205, 0]   # 0.2 s: too short
    out = drop_gk_hold(m, rates())
    assert np.isnan(out["bx"][50:80]).all()
    assert not np.isnan(out["bx"][200:205]).any()


def test_drift_extrapolates_into_a_gap():
    m = toy_match(n=300)
    m["bx"][100:200] = np.nan
    out = ball_drift(m, np.random.RandomState(0), rates(drift_p=1.0, drift_s=(1.0, 1.0), jitter_share=0.0))
    v = m["bx"][99] - m["bx"][98]
    assert out["bx"][100:125] == pytest.approx(m["bx"][99] + v * np.arange(1, 26), abs=1e-5)
    assert np.isnan(out["bx"][125:200]).all()


def test_fragmentation_keeps_every_position_and_team():
    m = toy_match(n=6000)
    out = fragment_players(m, np.random.RandomState(0), rates(piece_drop=0.0, piece_median_s=10.0))
    assert out["px"].shape[1] == 46 and out["gk"] == {0: 0, 1: 23}
    for t in (0, 1):
        a = np.sort(m["px"][:, m["slot_team"] == t], axis=1)
        b = np.sort(out["px"][:, out["slot_team"] == t], axis=1)[:, :11]
        assert np.allclose(a, b)
    assert np.allclose(out["px"][:, out["gk"][0]], m["px"][:, 0])     # keepers keep one slot


def test_double_count_adds_a_nearby_copy():
    m = toy_match(n=500)
    m["px"] = np.concatenate([m["px"], np.full((500, 24), np.nan, np.float32)], axis=1)
    m["py"] = np.concatenate([m["py"], np.full((500, 24), np.nan, np.float32)], axis=1)
    m["slot_team"] = np.r_[m["slot_team"], np.zeros(12), np.ones(12)].astype(np.int8)
    out = double_count(m, np.random.RandomState(0), rates(dup_share=1.0))
    present = (~np.isnan(out["px"])).sum(axis=1)
    assert np.all(present == 23)
    L, W = m["pitch"]
    f = 250
    extra = np.flatnonzero(~np.isnan(out["px"][f, 22:]))[0] + 22
    d = np.hypot((m["px"][f, :22] - out["px"][f, extra]) * L, (m["py"][f, :22] - out["py"][f, extra]) * W)
    assert 0.5 <= np.nanmin(d) <= 1.5 + 1e-3


def test_keeper_freeze():
    m = toy_match(n=1000)
    m["px"][:, 0] = np.linspace(0.02, 0.15, 1000)
    out = freeze_keepers(m, np.random.RandomState(0), rates(gk_freeze_share=1.0, gk_freeze_mean_s=100.0))
    x = out["px"][:, 0]
    assert np.mean(np.diff(x) == 0) > 0.9


def test_degrade_deterministic():
    m = toy_match()
    a, b, c = degrade(m, seed=1), degrade(m, seed=1), degrade(m, seed=2)
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in ("px", "py", "bx", "by"))
    assert not np.array_equal(a["px"], c["px"], equal_nan=True)
    assert np.array_equal(a["labels"], m["labels"]) and a["name"] == "toy+deg"


def test_view_window_stays_on_the_pitch():
    m = toy_match(n=500)
    m["bx"][:] = 0.98
    m["px"][:, 5] = 0.65
    out = view_window(m, np.random.RandomState(0), rates(view_width=(0.4, 0.4, 0.4), view_ball_pos=(0.19, 0.19)))
    assert not np.isnan(out["px"][:, 5]).any()          # unclipped, the window would start at 0.904


def test_keeper_out_of_view_is_held_where_last_seen():
    m = toy_match(n=500)
    m["bx"][:] = 0.7
    m["px"][:, 0] = np.r_[np.full(100, 0.55), np.linspace(0.55, 0.05, 400)]   # walks out of the view
    out = view_window(m, np.random.RandomState(0), rates(view_width=(0.4, 0.4, 0.4), view_ball_pos=(0.5, 0.5)))
    x = out["px"][:, 0]
    assert not np.isnan(x).any()
    gone = np.flatnonzero(m["px"][:, 0] < 0.5)[0]
    assert np.all(x[gone:] == x[gone - 1])


def test_lose_ball_cuts_stretches_at_the_requested_rate():
    m = toy_match(n=25 * 60 * 40)                                   # 40 minutes
    out = lose_ball(m, np.random.RandomState(0), rates(loss_per_min=1.0, loss_median_s=4.0, loss_sigma=0.0))
    miss = np.isnan(out["bx"])
    e = np.diff(np.r_[0, miss.astype(np.int8), 0])
    starts, ends = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    assert 25 <= len(starts) <= 55                                  # ~40 losses, Poisson
    assert np.median(ends - starts) == 100                          # 4 s each when they do not touch
    assert lose_ball(m, np.random.RandomState(0), rates(loss_per_min=0.0))["bx"] is not None
    assert not np.isnan(lose_ball(m, np.random.RandomState(0), rates(loss_per_min=0.0))["bx"]).any()


def test_rates_for_a_source():
    assert rates_for("metrica1")["loss_per_min"] == 0.0          # its own tracking already loses the ball in play
    assert rates_for("DFL-MAT-J03WOH")["loss_per_min"] == DEFAULT_RATES["loss_per_min"] > 0
    assert rates_for("metrica2")["still_frames"] == DEFAULT_RATES["still_frames"]


def test_smooth_ball_inside_observed_runs_only():
    from mtah.degrade import smooth_ball
    n = 40
    bx = np.r_[np.zeros(10), np.ones(10), np.full(5, np.nan), np.full(15, 0.5)].astype(np.float32)
    m = {"bx": bx, "by": np.zeros(n, np.float32), "name": "t"}
    off = smooth_ball(m, {"smooth_frames": 0})
    assert np.array_equal(off["bx"], bx, equal_nan=True)            # off by default: P2 unchanged
    s = smooth_ball(m, {"smooth_frames": 5})["bx"]
    assert np.isnan(s[20:25]).all() and np.allclose(s[25:], 0.5)     # gaps stay gaps, nothing leaks across
    assert 0 < s[9] < s[10] < 1 and s[0] == 0 and s[19] == 1        # the step becomes a ramp, run edges kept


def test_detector_reports_about_the_measured_share_and_mostly_on_the_ball():
    from mtah.degrade import DETECTOR_RATES, add_detector
    n = 50000
    clean = {"bx": np.full(n, 0.5, np.float32), "by": np.full(n, 0.5, np.float32)}
    deg = {"px": np.full((n, 2), 0.2, np.float32), "py": np.full((n, 2), 0.2, np.float32),
           "half": np.ones(n, np.int8), "pitch": (105.0, 68.0)}
    d = add_detector(clean, deg, seed=0)
    seen = ~np.isnan(d["dx"])
    assert abs(seen.mean() - DETECTOR_RATES["report"]) < 0.05
    good = np.hypot((d["dx"][seen] - 0.5) * 105, (d["dy"][seen] - 0.5) * 68) < 1.5
    assert abs(good.mean() - DETECTOR_RATES["good"]) < 0.06


def test_harsh_rates_scale_the_usual_ones_like_p9e():
    from mtah.degrade import harsh_rates, rates_for
    u, h = rates_for("DFL-MAT-J03WMX"), harsh_rates("DFL-MAT-J03WMX")
    assert h["loss_per_min"] == u["loss_per_min"] * 1.5 and h["jitter_share"] == u["jitter_share"] * 1.5
    assert h["drift_p"] == u["drift_p"] * 1.5 and h["piece_drop"] == min(1.0, u["piece_drop"] * 1.5) and h["smooth_frames"] == 50
    assert harsh_rates("metrica")["loss_per_min"] == 0.0


def test_view_window_can_leave_players_alone():
    import numpy as np
    from mtah.degrade import DEFAULT_RATES, view_window
    n = 2000
    m = {"bx": np.full(n, 0.5), "by": np.full(n, 0.5), "px": np.tile([0.01, 0.99, 0.5], (n, 1)).astype(float),
         "py": np.full((n, 3), 0.5), "gk": {0: -1, 1: -1}}
    cut = view_window(m, np.random.RandomState(0), dict(DEFAULT_RATES, view_width=(0.3, 0.3, 0.3)))
    keep = view_window(m, np.random.RandomState(0), dict(DEFAULT_RATES, view_width=(0.3, 0.3, 0.3), view_drops_players=False))
    assert np.isnan(cut["px"][:, 0]).all() and not np.isnan(keep["px"]).any()
