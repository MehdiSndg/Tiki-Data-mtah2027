from mtah import config as config_mod
from mtah.pipeline import joint
from mtah.validation.splits import cache_tag


def test_metrica_games_in():
    assert joint.metrica_games_in(["J03WQQ", "metrica_2", "metrica_2_v2b", "metrica_3_dr_s11"]) == [2, 3]
    assert joint.metrica_games_in(["J03WQQ"]) == []


def test_metrica_booster_configs():
    a, b = config_mod.load("ev_ext_mb"), config_mod.load("ev_ext_met_mb")
    assert a["joint"]["metrica_booster"] == [1, 2, 3] and a["joint"]["external_only"]
    assert a["joint"]["matches"] == config_mod.load("ev_ext")["joint"]["matches"]
    assert cache_tag(a) == cache_tag(b) != cache_tag(config_mod.load("ev_ext"))
    assert b["adapt"]["booster_source"] == "ev_ext_mb"
