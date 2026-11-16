"""Gün 4 (rapor 5 §2.1): ball-free collective kinematics group 'ck'."""
import numpy as np
import pytest

from mtah import config
from mtah.features import build_features, common_groups
from mtah.paths import CACHE

COLS = {"ck_adv_signed", "ck_adv_mag", "ck_phi_H", "ck_phi_A", "ck_phi_max", "ck_cen_vel_cos",
        "ck_width_logratio", "ck_ball_exit", "ck_missing_s", "ck_missing_coherent", "ck_last_owner",
        "ck_restart_spot", "ck_gk_depth_H", "ck_gk_depth_A", "ck_axis_align"}


def test_ck_is_source_common():
    assert "ck" in common_groups(["core", "ck", "tags"])


@pytest.mark.skipif(not (CACHE / "idsse_J03WPY").is_dir(), reason="needs cache/idsse_J03WPY")
def test_ck_columns_ranges_and_slot_permutation_invariance():
    from mtah.data import TrackingData
    cfg = config.load("main")
    d = TrackingData.load("idsse_J03WPY")
    n = 20_000
    d = TrackingData(d.name, d.fps, d.pitch, d.frame[:n], d.period[:n], d.players[:n], d.ball[:n].copy(),
                     d.team, d.labels[:n], dict(d.meta))
    d.ball[5_000:6_000] = np.nan                                   # a ball gap
    f = build_features(d, cfg, None, groups=["ck"])
    assert COLS <= set(f.columns)
    played = d.period > 0
    for c in ("ck_phi_H", "ck_phi_A"):
        v = f[c].to_numpy()[played]
        assert np.nanmin(v) >= 0 and np.nanmax(v) <= 1.0 + 1e-6
    assert np.allclose(f["ck_adv_mag"], np.abs(f["ck_adv_signed"]), equal_nan=True)
    assert (f["ck_missing_s"].to_numpy()[5_999] > 39) and (f["ck_missing_s"].to_numpy()[6_001] == 0)
    rng = np.random.default_rng(0)
    perm = np.concatenate([rng.permutation(np.flatnonzero(d.team == 0)), rng.permutation(np.flatnonzero(d.team == 1))])
    rng.shuffle(perm)
    dp = TrackingData(d.name, d.fps, d.pitch, d.frame, d.period, d.players[:, perm], d.ball, d.team[perm],
                      d.labels, dict(d.meta))
    fp = build_features(dp, cfg, None, groups=["ck"])
    for c in sorted(COLS):
        tol = 1e-3 if c == "ck_axis_align" else 1e-5        # PCA of near-isotropic blocks: float32 noise
        assert np.allclose(f[c].to_numpy(), fp[c].to_numpy(), equal_nan=True, atol=tol), c
