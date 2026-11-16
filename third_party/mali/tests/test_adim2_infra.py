"""Label alignment, feature cache equivalence, per-match kick-rate calibration."""

import numpy as np
import pytest

from mtah import config
from mtah.constants import A, D, H
from mtah.data import TrackingData
from mtah.labels.align import suppress_short_returns
from mtah.paths import tactical


def test_short_return_spells_are_suppressed_but_real_ones_kept():
    lab = np.concatenate([np.full(30, A), np.full(5, H), np.full(30, A),     # touch -> A
                          np.full(20, H), np.full(40, A),                    # 20-frame H is real
                          np.full(6, H), np.full(50, D), np.full(30, H)])    # next to D: kept
    out = suppress_short_returns(lab, [(0, len(lab))], max_frames=11)
    assert np.all(out[:65] == A)
    assert np.all(out[65:85] == H)
    assert np.all(out[125:131] == H) and np.all(out[131:181] == D)


def _head(data, n):
    return TrackingData(data.name, data.fps, data.pitch, data.frame[:n], data.period[:n],
                        data.players[:n], data.ball[:n], data.team,
                        None if data.labels is None else data.labels[:n], dict(data.meta))


@pytest.mark.skipif(not tactical("atd_raw").is_file(), reason="match data not present")
def test_feature_cache_returns_identical_values(tmp_path, monkeypatch):
    from mtah.features import registry
    from mtah.io.atd import cached_barath
    monkeypatch.setattr(registry, "CACHE", tmp_path)
    cfg = config.load("m0")
    data = _head(cached_barath(), 25_000)
    fresh = registry.build_features(data, cfg, None, groups=["core", "kin"], cache_key="t")
    again = registry.build_features(data, cfg, None, groups=["core", "kin"], cache_key="t")
    assert len(list((tmp_path / "features" / "t").glob("*.parquet"))) == 2
    assert fresh.columns.equals(again.columns)
    assert np.array_equal(fresh.to_numpy(), again.to_numpy(), equal_nan=True)


@pytest.mark.skipif(not tactical("atd_raw").is_file(), reason="match data not present")
def test_kick_rate_is_calibrated_per_match():
    if not TrackingData.exists("idsse_J03WPY"):
        pytest.skip("IDSSE J03WPY not converted")
    from mtah.features import build_features
    from mtah.io.atd import cached_barath
    cfg = config.load("m0")
    target = 12.0
    for data in (_head(cached_barath(), 60_000), _head(TrackingData.load("idsse_J03WPY"), 60_000)):
        f = build_features(data, cfg, None, groups=["kin"])
        live = (data.period > 0) & ~np.isnan(data.ball[:, 0])
        rate = f["kin_kick"].to_numpy()[data.period > 0].sum() / (live.sum() / data.fps / 60)
        assert abs(rate - target) < 1.0, (data.name, rate)
