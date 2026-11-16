"""Gün 3 (Ek L.14): source-specific camera features from the tactical homography."""
import numpy as np
import pytest

from mtah import config
from mtah.features import build_features, common_groups
from mtah.paths import CACHE, tactical


def test_cam_group_is_source_specific():
    assert "cam" not in common_groups(["core", "geo", "kin", "cam", "tags"])


@pytest.mark.skipif(not tactical("homography").is_file() or not (CACHE / "barath").is_dir(),
                    reason="needs the BAR-ATH homography and cache/barath")
def test_cam_aim_tracks_the_ball_and_pan_is_finite():
    from mtah.data import TrackingData
    from mtah.io.tags import load_tags
    cfg = config.load("d3_cam")
    bar = TrackingData.load("barath")
    f = build_features(bar, cfg, load_tags(), groups=["cam"])
    assert {"cam_xc", "cam_ball_dx", "cam_ball_adx", "cam_pan"} <= set(f.columns)
    played = bar.period > 0
    vis = ~np.isnan(bar.ball[:, 0]) & played
    ball_xc = np.where(build_features(bar, cfg, load_tags(), groups=["core"])["ball_xc"].notna(),
                       build_features(bar, cfg, load_tags(), groups=["core"])["ball_xc"], np.nan)
    r = np.corrcoef(f["cam_xc"].to_numpy()[vis], ball_xc[vis])[0, 1]
    assert r > 0.7
    assert np.isfinite(f["cam_pan"].to_numpy()[played]).mean() > 0.99
    with pytest.raises(ValueError):
        build_features(bar, cfg, None, groups=["cam"])          # source-specific input check
