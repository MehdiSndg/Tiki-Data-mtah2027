"""Gün 4 (Ek M.10): touch-sequence group 'tc'."""
import numpy as np
import pytest

from mtah import config
from mtah.features import build_features, common_groups
from mtah.paths import CACHE

COLS = {"tc_last_team", "tc_next_team", "tc_last2_team", "tc_run_prev", "tc_run_next", "tc_run_start_since",
        "tc_next_run_to", "tc_switch_pending", "tc_switches_3s", "tc_events_3s", "tc_margin_last",
        "tc_margin_next", "tc_close_H", "tc_close_A", "tc_close_diff", "tc_duel"}


def test_tc_is_source_common():
    assert "tc" in common_groups(["core", "tc", "tags"])


@pytest.mark.skipif(not (CACHE / "idsse_J03WPY").is_dir(), reason="needs cache/idsse_J03WPY")
def test_tc_columns_consistency_and_slot_permutation_invariance():
    from mtah.data import TrackingData
    cfg = config.load("main")
    d = TrackingData.load("idsse_J03WPY")
    n = 20_000
    d = TrackingData(d.name, d.fps, d.pitch, d.frame[:n], d.period[:n], d.players[:n], d.ball[:n].copy(),
                     d.team, d.labels[:n], dict(d.meta))
    d.ball[5_000:6_000] = np.nan
    f = build_features(d, cfg, None, groups=["tc"])
    assert COLS <= set(f.columns)
    played = d.period > 0
    for c in ("tc_last_team", "tc_next_team", "tc_last2_team"):
        assert set(np.unique(f[c].to_numpy()[played])) <= {-1.0, 0.0, 1.0}, c
    rp = f["tc_run_prev"].to_numpy()
    assert rp.min() >= 0 and rp.max() <= 6
    sw = f["tc_switch_pending"].to_numpy()
    lt, nt = f["tc_last_team"].to_numpy(), f["tc_next_team"].to_numpy()
    assert np.array_equal(sw == 1, (lt != 0) & (nt != 0) & (lt != nt))
    assert (f["tc_events_3s"].to_numpy()[played] > 0).mean() > 0.3       # touches are frequent in play
    assert np.isnan(f["tc_close_H"].to_numpy()[5_500])                    # hidden ball -> NaN
    # per-match calibrated events: roughly kick + reception rate per visible minute
    rng = np.random.default_rng(0)
    perm = np.concatenate([rng.permutation(np.flatnonzero(d.team == 0)), rng.permutation(np.flatnonzero(d.team == 1))])
    rng.shuffle(perm)
    dp = TrackingData(d.name, d.fps, d.pitch, d.frame, d.period, d.players[:, perm], d.ball, d.team[perm],
                      d.labels, dict(d.meta))
    fp = build_features(dp, cfg, None, groups=["tc"])
    for c in sorted(COLS):
        assert np.allclose(f[c].to_numpy(), fp[c].to_numpy(), equal_nan=True, atol=1e-5), c
