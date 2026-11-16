"""Test-time team-exchange averaging (Rapor 9 Sıra 1): synthetic checks only, no training."""

import numpy as np
import pytest

from mtah import config
from mtah.augmentation import SWAP, swapped_steps, tta_boundary, tta_probs
from mtah.constants import A, D, H
from mtah.features import common_groups
from mtah.pipeline import adapt

from test_team_exchange import toy


def random_probs(n, seed):
    p = np.random.default_rng(seed).random((n, 3)) + 0.05
    return p / p.sum(1, keepdims=True)


def test_swap_is_an_involution_on_hda_columns():
    assert SWAP == [A, H, D]
    p = random_probs(50, 0)
    np.testing.assert_array_equal(p[:, SWAP][:, SWAP], p)


def test_equivariant_model_is_unchanged_by_averaging():
    """A model that already obeys the symmetry (output on the exchanged view = permuted output)."""
    p = random_probs(200, 1)
    np.testing.assert_allclose(tta_probs(p, p[:, SWAP]), p, rtol=1e-9, atol=1e-12)


def test_average_is_a_normalised_geometric_mean_and_symmetric_in_views():
    p, q = random_probs(300, 2), random_probs(300, 3)
    t = tta_probs(p, q)
    np.testing.assert_allclose(t.sum(1), 1.0)
    g = np.sqrt(p * q[:, SWAP])
    np.testing.assert_allclose(t, g / g.sum(1, keepdims=True), rtol=1e-9)
    # exchanging which view is called "original" gives the exchanged answer
    np.testing.assert_allclose(tta_probs(q, p), t[:, SWAP], rtol=1e-9)
    # the averaged prediction is exactly equivariant: P(H) - P(A) flips sign under the exchange
    np.testing.assert_allclose(tta_probs(q, p)[:, H], t[:, A], rtol=1e-9)
    np.testing.assert_allclose(t[:, D], tta_probs(q, p)[:, D], rtol=1e-9)


def test_boundary_average_keeps_zero_outside_runs_and_decoder_rounding():
    b1 = np.r_[np.zeros(5), np.linspace(0.1, 0.9, 10), np.zeros(5)]
    b2 = np.r_[np.zeros(5), np.linspace(0.3, 0.5, 10), np.zeros(5)]
    out = tta_boundary(b1, b2)
    assert np.all(out[:5] == 0) and np.all(out[-5:] == 0)
    np.testing.assert_array_equal(out, np.round(out, 6))
    np.testing.assert_allclose(out[5:15], np.round((b1[5:15] + b2[5:15]) / 2, 6))


def test_inference_view_has_no_targets_and_permuted_booster_input(monkeypatch):
    from mtah import features
    data = toy()
    cfg = config.load("ev_ha_base_rf0")
    original = features.build_features
    monkeypatch.setattr(features, "build_features", lambda data, cfg, **kw: original(data, cfg, groups=kw.get("groups")))
    feats = original(data, cfg, groups=common_groups(cfg["features"]["groups"]))
    probs = np.tile([0.6, 0.3, 0.1], (data.n_frames, 1))
    prep = {"corrupted": data, "runs": data.runs(), "key": "synthetic", "feats": feats}
    steps = swapped_steps(prep, cfg, list(feats.columns), probs, None, 5)
    assert steps
    for x, y, known, _ in steps:
        assert not known.any() and not y.any()
        np.testing.assert_allclose(x[:, len(feats.columns):len(feats.columns) + 3],
                                   np.tile(np.log([0.3, 0.6, 0.1]), (len(x), 1)), rtol=1e-6)


def test_default_is_off_and_official_recipe_unchanged():
    assert adapt.DEFAULTS["tta_swap"] is None
    ac, _ = adapt.params(config.load("main_ha"))
    assert ac.get("tta_swap") is None


def test_screen_configs_differ_from_official_only_by_tta_and_fresh_seeds():
    for region in (0, 1):
        cfg, ref = config.load(f"ev_tta_rf{region}"), config.load(f"main_ha_rf{region}")
        assert cfg["adapt"]["tta_swap"] == "report" and cfg["adapt"]["net_seed_offset"] == 40000
        assert cfg["validation"] == ref["validation"] and cfg["seqnet"] == ref["seqnet"]
        for k in ("tta_swap", "net_seed_offset", "eval_matches", "segment_diagnostics", "booster_source"):
            cfg["adapt"].pop(k, None)
            ref["adapt"].pop(k, None)
        assert cfg["adapt"] == ref["adapt"]


def fake_scores(v1, v2b, met, per_v2b, per_met, poss=240, dead=0.45, home=0.61):
    return dict(dfl_v1=v1, dfl_v2b=v2b, metrica=met,
                dfl_v2b_per_match=dict(zip(("a", "b", "c"), per_v2b)),
                metrica_per_match=dict(zip(("m1", "m2", "m3"), per_met)),
                bar=dict(n_possessions=poss, dead_share=dead, home_share=home))


def test_four_gates_pass_and_each_failure(monkeypatch):
    from test_team_exchange import day_runner
    runner = day_runner(monkeypatch)
    ref = fake_scores(.72, .70, .65, (.7, .7, .7), (.65, .65, .65))
    good = fake_scores(.725, .71, .66, (.71, .71, .69), (.66, .66, .66))
    res = runner.four_gates({"rf0": {"reference": ref, "candidate": good}})
    assert res["pass"] and res["positive_matches"] == 5
    assert res["delta_composite"] == pytest.approx(.45 * .01 + .35 * .01 + .2 * .005)
    weak_v1 = dict(good, dfl_v1=.71)
    assert not runner.four_gates({"rf0": {"reference": ref, "candidate": weak_v1}})["gates"]["2_non_inferiority"]
    signs = fake_scores(.725, .71, .66, (.75, .69, .69), (.70, .64, .64))
    assert not runner.four_gates({"rf0": {"reference": ref, "candidate": signs}})["gates"]["3_match_signs"]
    smooth = fake_scores(.725, .71, .66, (.71, .71, .69), (.66, .66, .66), poss=215)
    assert not runner.four_gates({"rf0": {"reference": ref, "candidate": smooth}})["gates"]["4_label_free"]
    tiny = fake_scores(.721, .701, .651, (.701, .701, .701), (.651, .651, .651))
    assert not runner.four_gates({"rf0": {"reference": ref, "candidate": tiny}})["gates"]["1_composite"]


def test_tta_comparison_pairs_inside_each_run(monkeypatch):
    from test_team_exchange import day_runner
    runner = day_runner(monkeypatch)
    monkeypatch.setattr(runner, "scores", lambda value: value["s"])
    ref = fake_scores(.72, .70, .65, (.7, .7, .7), (.65, .65, .65))
    cand = fake_scores(.73, .71, .66, (.71, .71, .71), (.66, .66, .66))
    data = {f"ev_tta_rf{r}": {"s": ref, "tta": {"s": cand}} for r in (0, 1)}
    out = runner.compare("tta", data, "synthetic")
    assert out["mean_delta"]["dfl_v2b"] == pytest.approx(.01) and out["four_gates"]["pass"]


def test_symmetric_anchor_screen_matches_tta_reference_except_twins():
    assert adapt.DEFAULTS["bar_swap_anchors"] is False
    for region in (0, 1):
        sym, ref = config.load(f"ev_symft_rf{region}"), config.load(f"ev_tta_rf{region}")
        assert sym["adapt"].pop("bar_swap_anchors") is True and ref["adapt"].pop("tta_swap") == "report"
        assert sym["adapt"] == ref["adapt"] and sym["seqnet"] == ref["seqnet"] and sym["validation"] == ref["validation"]


def test_symmetric_confirmation_pairs_differ_only_by_twins():
    for region in (0, 1):
        base, sym = config.load(f"ev_symft_confirm_base_rf{region}"), config.load(f"ev_symft_confirm_sym_rf{region}")
        assert sym["adapt"].pop("bar_swap_anchors") is True and base["adapt"]["net_seed_offset"] == 50000
        base.pop("name"), sym.pop("name")
        assert base == sym
