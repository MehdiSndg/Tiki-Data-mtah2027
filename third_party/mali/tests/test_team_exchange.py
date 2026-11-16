"""Synthetic-only team exchange and daytime scheduling checks; no real match/GPU training."""

from datetime import datetime, timedelta, timezone
import importlib
from pathlib import Path
import subprocess

import numpy as np
import pytest

from mtah import config
from mtah.augmentation import swap_labels, swap_tracking, training_steps
from mtah.constants import H, A, D, X
from mtah.data import TrackingData
from mtah.features import build_features, common_groups
from mtah.features.context import FeatureContext


def toy():
    n = 500
    players = np.zeros((n, 6, 2), np.float32)
    players[:, :, 0] = [4, 35, 44, 101, 65, 56]
    players[:, :, 1] = [34, 20, 45, 34, 20, 45]
    players[:, :, 0] += np.sin(np.arange(n)[:, None] / 30)
    period = np.r_[np.ones(n // 2), np.full(n // 2, 2)].astype(np.int8)
    players[n // 2:, :, 0] = 105 - players[n // 2:, :, 0]
    ball = np.c_[45 + 12 * np.sin(np.arange(n) / 21), np.full(n, 34)].astype(np.float32)
    ball[50:60] = np.nan
    return TrackingData("synthetic", 25, (105, 68), np.arange(n), period, players, ball,
                        np.array([H, H, H, A, A, A], np.int8), np.resize([H, A, D, X], n).astype(np.int8),
                        {"home": "one", "away": "two", "source": "synthetic"})


def test_exchange_is_involution_preserves_geometry_unknown_dead_and_source():
    data = toy()
    swapped = swap_tracking(data)
    restored = swap_tracking(swapped)
    assert swapped.name == data.name and swapped.meta["home"] == "two"
    assert swapped.players is data.players and swapped.ball is data.ball
    assert swapped.frame is data.frame and swapped.period is data.period
    assert restored.meta == data.meta
    np.testing.assert_array_equal(restored.team, data.team)
    np.testing.assert_array_equal(restored.labels, data.labels)
    np.testing.assert_array_equal(swap_labels(np.array([H, A, D, X, -1])), [A, H, D, X, -1])
    cfg = config.load("ev_ha_base_rf0")
    old, new = FeatureContext(data, cfg), FeatureContext(swapped, cfg)
    for period, (gh, ga, direction) in old.gk_dir.items():
        assert new.gk_dir[period] == (ga, gh, not direction)
    np.testing.assert_allclose(old.d_home, new.d_away, equal_nan=True)
    finite = np.isfinite(data.ball[:, 0])
    np.testing.assert_allclose((old.xc_b + new.xc_b)[finite], 105)


def test_full_feature_recomputation_recovers_after_two_exchanges():
    data = toy()
    cfg = config.load("ev_ha_base_rf0")
    groups = common_groups(cfg["features"]["groups"])
    original = build_features(data, cfg, groups=groups)
    restored = build_features(swap_tracking(swap_tracking(data)), cfg, groups=groups)
    exchanged = build_features(swap_tracking(data), cfg, groups=groups)
    np.testing.assert_allclose(original.values, restored.values, equal_nan=True)
    np.testing.assert_allclose(original["prox_dH"], exchanged["prox_dA"], equal_nan=True)
    assert original.shape == exchanged.shape


@pytest.mark.parametrize("include_tokens", [False, True])
def test_swapped_step_inputs_permute_booster_and_targets_without_new_corruption(monkeypatch, include_tokens):
    from mtah import augmentation, features
    from mtah.pipeline.hybrid import step_view
    data = toy()
    cfg = config.load("ev_ha_base_rf0")
    original = features.build_features
    # Exercise real feature computation while avoiding any on-disk cache in this synthetic test.
    monkeypatch.setattr(features, "build_features", lambda data, cfg, **kw:
                        original(data, cfg, groups=kw.get("groups")))
    feats = original(data, cfg, groups=common_groups(cfg["features"]["groups"]))
    probs = np.tile([0.6, 0.3, 0.1], (data.n_frames, 1))
    prep = {"corrupted": data, "runs": data.runs(), "key": "synthetic", "feats": feats}
    swapped = augmentation.swapped_steps(prep, cfg, list(feats.columns), probs, data.labels, 5,
                                         token_params={"token_pool":"deepsets_team"} if include_tokens else None)
    reference = step_view(feats, list(feats.columns), probs, data.labels, data.runs(), 5)
    for (x, y, known, run), (_, yr, kr, rr) in zip(swapped, reference):
        assert run == rr
        np.testing.assert_array_equal(known, kr)
        np.testing.assert_array_equal(y[known], swap_labels(yr[known]))
        np.testing.assert_allclose(x[:, len(feats.columns):len(feats.columns)+3],
                                   np.tile(np.log([0.3, 0.6, 0.1]), (len(x), 1)), rtol=1e-6)
        if include_tokens:
            tok = x[:, -216:].reshape(len(x),24,9)
            assert np.all(tok[..., -1].sum(1) == 6)
            assert np.all((tok[..., -1] * tok[..., 5]).sum(1) == 2)


def test_outer_fold_removes_primary_mixed_and_all_twins_and_disabled_order_is_unchanged():
    primary = {"one": ["p1"], "two": ["p2"]}
    mixed = {"one": ["m1"], "two": ["m2"]}
    copies = {"one": ["p1swap", "m1swap"], "two": ["p2swap", "m2swap"]}
    assert training_steps(list(primary), primary, mixed) == ["p1", "p2", "m1", "m2"]
    assert training_steps(list(primary), primary, mixed, copies, "one") == ["p2", "m2", "p2swap", "m2swap"]


def test_configs_isolate_copies_keep_referee_blind_and_share_booster():
    from mtah.validation.splits import cache_tag
    rows = [config.load(n) for n in ("ev_ha_base_rf0", "ev_ha_repeat_rf0", "ev_ha_swap_rf0")]
    for row in rows:
        assert row["seqnet"]["n_seeds"] == 2 and row["adapt"]["booster_source"] == "main_rf0"
        assert row["adapt"]["metrica_eval"] == [1, 2, 3] and not row["adapt"].get("short_team")
        assert not set(row["adapt"]["eval_matches"]) & set(row["joint"]["matches"])
        assert cache_tag(row) == cache_tag(config.load("main_rf0"))
    for row in rows:
        row.pop("name")
        row["adapt"].pop("external_team_copies", None)
    assert rows[0] == rows[1] == rows[2]


def day_runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("run_day_screen")


def test_no_fit_can_start_when_its_budget_crosses_cutoff(monkeypatch):
    runner = day_runner(monkeypatch)
    tz = timezone(timedelta(hours=3))
    cutoff = datetime(2026, 10, 28, 22, 55, tzinfo=tz)
    assert runner.budget_seconds(cutoff - timedelta(hours=2), cutoff, 5400) == 5400
    assert runner.budget_seconds(cutoff - timedelta(minutes=94), cutoff, 5400) == 0
    assert runner.budget_seconds(cutoff + timedelta(minutes=1), cutoff, 1) == 0


def test_timeout_stops_only_the_owned_child_tree(monkeypatch, tmp_path):
    runner = day_runner(monkeypatch)
    calls = []
    class Child:
        pid = 1234
        def poll(self):
            return None
        def wait(self, timeout):
            if timeout == 2:
                raise subprocess.TimeoutExpired("synthetic", timeout)
            return 1
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    with pytest.raises(subprocess.TimeoutExpired):
        runner.fit(["synthetic"], tmp_path, {}, tmp_path / "log", 2)
    assert calls == [["taskkill", "/PID", "1234", "/T", "/F"]]


def test_confirmation_seeds_are_fresh_paired_and_do_not_change_booster_identity():
    from mtah.pipeline.adapt import network_seeds, params
    from mtah.validation.splits import cache_tag
    first = config.load("ev_ha_base_rf0")
    first_ac, first_p = params(first)
    old = set(network_seeds(first, first_ac, first_p))
    for region in (0, 1):
        base = config.load(f"ev_ha_confirm_base_rf{region}")
        swap = config.load(f"ev_ha_confirm_swap_rf{region}")
        for cfg in (base, swap):
            ac, p = params(cfg)
            seeds = network_seeds(cfg, ac, p)
            assert seeds == [12027, 13027, 14027, 15027, 16027] and not old & set(seeds)
            assert cfg["seed"] == 2027 and cfg["validation"]["region_cv"] == [region, 4]
            assert cache_tag(cfg) == cache_tag(config.load(f"main_rf{region}"))
            assert cfg["adapt"]["booster_source"] == f"main_rf{region}"
            assert cfg["adapt"]["metrica_eval"] == [1, 2, 3] and not cfg["adapt"].get("short_team")
        base.pop("name")
        swap.pop("name")
        assert swap["adapt"].pop("external_team_copies") == "swap"
        assert base == swap


def test_confirmation_comparison_preserves_region_pairs(monkeypatch):
    runner = day_runner(monkeypatch)
    monkeypatch.setattr(runner, "scores", lambda value: value)
    data = {}
    for region, delta in ((0, .01), (1, -.02)):
        data[f"ev_ha_confirm_base_rf{region}"] = dict(dfl_v1=.70, dfl_v2b=.68, metrica=.65)
        data[f"ev_ha_confirm_swap_rf{region}"] = dict(dfl_v1=.70 + delta, dfl_v2b=.68 + delta, metrica=.65 + delta)
    result = runner.compare("ha-confirm", data, "synthetic")
    assert set(result["pairs"]) == {"rf0", "rf1"}
    assert result["pairs"]["rf0"]["delta"]["metrica"] == pytest.approx(.01)
    assert result["pairs"]["rf1"]["delta"]["metrica"] == pytest.approx(-.02)
    assert result["mean_delta"]["metrica"] == pytest.approx(-.005)
