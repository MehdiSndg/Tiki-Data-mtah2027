"""Feature groups share one schema across sources; source-specific groups are fenced off."""

import numpy as np
import pytest

from mtah import config
from mtah.data import TrackingData
from mtah.features import build_features, common_groups
from mtah.paths import tactical

ALL = ["core", "tags", "kin", "geo", "press"]


def _head(data, n):
    """First n frames of a TrackingData (fast tests)."""
    return TrackingData(data.name, data.fps, data.pitch, data.frame[:n], data.period[:n],
                        data.players[:n], data.ball[:n], data.team,
                        None if data.labels is None else data.labels[:n], dict(data.meta))


def test_common_groups_exclude_source_specific():
    assert common_groups(ALL) == ["core", "kin", "geo", "press"]


@pytest.mark.skipif(not tactical("atd_raw").is_file(), reason="match data not present")
def test_same_schema_on_barath_and_idsse():
    if not TrackingData.exists("idsse_J03WPY"):
        pytest.skip("IDSSE J03WPY not converted")
    from mtah.io.atd import cached_barath
    cfg = config.load("m0")
    bar = _head(cached_barath(), 30_000)
    dfl = _head(TrackingData.load("idsse_J03WPY"), 20_000)
    fb = build_features(bar, cfg, None, groups=common_groups(ALL))
    fd = build_features(dfl, cfg, None, groups=common_groups(ALL))
    assert list(fb.columns) == list(fd.columns)
    assert all(fb.dtypes == np.float32) and all(fd.dtypes == np.float32)
    with pytest.raises(ValueError):
        build_features(dfl, cfg, None, groups=["tags"])


def test_paired_bootstrap_sanity():
    from mtah.constants import A, D, H, N_FRAMES, PUBLIC_WINDOW, X
    from mtah.validation.compare import paired_spell_bootstrap
    gt = np.full(N_FRAMES, X, np.int8)
    a0, a1 = PUBLIC_WINDOW
    pattern = np.repeat([H, A, D, H, A, H], (a1 - a0) // 6 + 1)[: a1 - a0]
    gt[a0:a1] = pattern
    perfect = gt.copy()
    flat = gt.copy()
    flat[a0:a1] = H
    same = paired_spell_bootstrap(gt, perfect, perfect, n_draws=200)
    assert same["delta"] == 0 and same["p_b_better"] == 0
    better = paired_spell_bootstrap(gt, flat, perfect, n_draws=200)
    assert better["delta"] > 0 and better["p_b_better"] > 0.9
