"""DFL -> ATD-like ball degradation (pipeline/dfl_referee.atd_like_ball, Ek M.7)."""
import numpy as np

from mtah.pipeline import dfl_referee


def _track(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    b = np.c_[20 + 0.2 * t, 30 + 5 * np.sin(t / 30)] + rng.normal(0, 0.1, (n, 2))
    b[100:150] = np.nan          # a gap
    b[300:305] = np.nan
    return b.astype(np.float32)


def _dv(b):
    step = np.linalg.norm(np.diff(b, axis=0), axis=1)
    return np.nanmedian(np.abs(np.diff(step)))


def test_smoothing_reduces_jitter_and_keeps_gaps():
    b = _track()
    out = dfl_referee.atd_like_ball(b, 13, 0.0, 25, np.random.default_rng(1))
    assert out.dtype == np.float32 and out.shape == b.shape
    assert np.array_equal(np.isnan(out), np.isnan(b))
    assert _dv(out) < 0.5 * _dv(b)


def test_short_runs_untouched_and_no_op_defaults():
    b = _track()
    b[150:160] = 5.0             # a 10-frame visible run between gaps is shorter than the window
    b[160:300] = np.nan
    out = dfl_referee.atd_like_ball(b, 13, 0.0, 25, np.random.default_rng(1))
    assert np.allclose(out[150:160], b[150:160])
    same = dfl_referee.atd_like_ball(b, 0, 0.0, 25, np.random.default_rng(1))
    assert np.allclose(same, b, equal_nan=True)


def test_offset_is_slow_scaled_and_deterministic():
    b = _track(4000)
    o1 = dfl_referee.atd_like_ball(b, 0, 0.5, 25, np.random.default_rng(3))
    o2 = dfl_referee.atd_like_ball(b, 0, 0.5, 25, np.random.default_rng(3))
    assert np.array_equal(o1, o2, equal_nan=True)
    d = (o1 - b)[~np.isnan(b[:, 0])]
    assert 0.35 < d.std(axis=0).mean() < 0.65
    # slowly varying: frame-to-frame offset change is tiny compared with its size
    assert np.nanmedian(np.abs(np.diff(o1 - b, axis=0))) < 0.05


def test_default_referee_cfg_keeps_ball_untouched():
    rc = dfl_referee.referee_cfg({})
    assert rc["ball_smooth_frames"] == 0 and rc["ball_offset_m"] == 0.0
