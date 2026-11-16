"""Organisers' Nov 2026 clarification: training / tuning on external data only (no provided-game training)."""

from mtah import config
from mtah.pipeline import joint
from mtah.validation.splits import cache_tag


class Bar:
    name = "barath"
    share = None


def test_external_only_sources_never_include_the_provided_game(monkeypatch):
    cfg = config.load("ev_ext")
    monkeypatch.setattr(joint, "dfl_sources", lambda c, m, l, tp=None, share_total=None: [("dfl", m, share_total)])
    out = joint.training_sources(cfg, Bar(), ["J03WPY"], None)
    assert out == [("dfl", ["J03WPY"], 1.0)]
    legacy = config.load("main_ha")
    bar = Bar()
    out = joint.training_sources(legacy, bar, ["J03WPY"], None)
    assert out[0] is bar and bar.share == joint.joint_cfg(legacy)["bar_share"]


def test_external_recipe_has_own_caches_no_tags_and_no_bar_options():
    ext, ha = config.load("main_ext"), config.load("main_ha")
    assert ext["joint"]["external_only"] is True and "tags" not in ext["features"]["groups"]
    assert cache_tag(ext) != cache_tag(ha)
    assert not ext["adapt"].get("bar_swap_anchors") and not ext["adapt"].get("tta_swap")
    assert ext["adapt"]["external_team_copies"] == "swap" and ext.get("dfl_mix_eval")
    assert "booster_source" not in ext["adapt"]


def test_official_recipe_unchanged_by_the_switch():
    assert joint.JOINT_DEFAULTS["external_only"] is False
    assert not config.load("main_ha").get("joint", {}).get("external_only")


def test_metrica_training_recipe_scores_metrica_leave_one_out_and_keeps_dfl_referees():
    from mtah.pipeline import adapt
    cfg = config.load("ev_ext_met")
    ac, _ = adapt.params(cfg)
    assert cfg["joint"]["external_only"] and ac["metrica_corrupt"] and not ac["metrica_eval"]
    assert set(ac["eval_matches"]) == {"J03WQQ", "J03WOY", "J03WR9", "metrica_1", "metrica_2", "metrica_3"}
    ref = config.load("ev_ext")
    assert adapt.network_seeds(cfg, ac, adapt.params(cfg)[1]) == adapt.network_seeds(ref, *adapt.params(ref))
    assert "metrica" not in str(cfg["joint"]["matches"])        # booster never sees Metrica


def test_training_steps_drop_every_copy_of_the_held_out_game():
    from mtah.augmentation import training_steps
    out = training_steps(["a", "metrica_1"], {"a": [1], "metrica_1": [2]}, {"a": [3], "metrica_1": [4]},
                         {"a": [5], "metrica_1": [6]}, exclude="metrica_1")
    assert out == [1, 3, 5]
