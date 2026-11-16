"""H<->A production candidate: same booster as the official main, only the net's team-copy flag differs."""
from mtah import config
from mtah.pipeline import regionrun
from mtah.validation.splits import cache_tag


def test_main_ha_folds_mirror_official_main():
    ha, main = regionrun.fold_configs("main_ha"), regionrun.fold_configs("main")
    assert [[n for n, _ in r] for r in ha] == [[n.replace("main", "main_ha", 1) for n, _ in r] for r in main]
    for row_ha, row_main in zip(ha, main):
        for (_, a), (_, b) in zip(row_ha, row_main):
            assert a["adapt"]["external_team_copies"] == "swap" and not b["adapt"].get("external_team_copies")
            assert a["seed"] == b["seed"] and cache_tag(a) == cache_tag(b)
            a2 = dict(a, name=b["name"], adapt=dict(a["adapt"]))
            a2["adapt"].pop("external_team_copies")
            assert config.digest(a2) == config.digest(b)
