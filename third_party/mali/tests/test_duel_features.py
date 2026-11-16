"""Gün 5 (Ek M.12): duel dipole / point-wise pitch-control group 'dl'."""
import numpy as np
import pytest

from mtah import config
from mtah.features import build_features, common_groups
from mtah.paths import CACHE

COLS = {"dl_dist_z_H", "dl_dist_z_A", "dl_dist_z_diff", "dl_close_diff", "dl_out_align_diff", "dl_deflect",
        "dl_shield", "dl_tti_now", "dl_ppcf_1s", "dl_ppcf_gain", "dl_event_team"}


def test_dl_is_source_common():
    assert "dl" in common_groups(["core", "dl", "tags"])


@pytest.mark.skipif(not (CACHE / "idsse_J03WPY").is_dir(), reason="needs cache/idsse_J03WPY")
def test_dl_ranges_hidden_ball_and_slot_permutation_invariance():
    from mtah.data import TrackingData
    cfg = config.load("main")
    d = TrackingData.load("idsse_J03WPY")
    n = 20_000
    d = TrackingData(d.name, d.fps, d.pitch, d.frame[:n], d.period[:n], d.players[:n], d.ball[:n].copy(),
                     d.team, d.labels[:n], dict(d.meta))
    d.ball[5_000:6_000] = np.nan
    f = build_features(d, cfg, None, groups=["dl"])
    assert COLS <= set(f.columns)
    assert f.loc[5_500, list(COLS)].isna().all()
    for c in ("dl_tti_now", "dl_ppcf_1s"):
        v = f[c].to_numpy()
        assert np.nanmin(v) >= -1 - 1e-9 and np.nanmax(v) <= 1 + 1e-9
    assert np.nanmax(np.abs(f["dl_dist_z_H"].to_numpy())) <= 5.0
    rng = np.random.default_rng(0)
    perm = np.concatenate([rng.permutation(np.flatnonzero(d.team == 0)), rng.permutation(np.flatnonzero(d.team == 1))])
    rng.shuffle(perm)
    dp = TrackingData(d.name, d.fps, d.pitch, d.frame, d.period, d.players[:, perm], d.ball, d.team[perm],
                      d.labels, dict(d.meta))
    fp = build_features(dp, cfg, None, groups=["dl"])
    for c in sorted(COLS):
        assert np.allclose(f[c].to_numpy(), fp[c].to_numpy(), equal_nan=True, atol=1e-6), c
