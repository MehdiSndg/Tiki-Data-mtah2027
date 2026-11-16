import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))


def test_cache_path_tells_targets_apart():
    from mtah.members_c import cache_path
    assert cache_path("team", "gru", False, (0, 1), 1) != cache_path("boundary", "gru", False, (0, 1), 1)


def test_cache_path_tells_held_out_matches_apart():
    from mtah.members_c import cache_path
    assert cache_path("team", "gru", False, (0, 1), 1, exclude="J03WMX") != cache_path("team", "gru", False, (0, 1), 1)


def test_cache_path_is_recipe_c_not_production():
    from mtah.members_c import cache_path
    assert "_gruC" in os.path.basename(cache_path("team", "big", True, range(6), 1))


def test_cache_path_tells_ball_smoothing_apart():
    from mtah.members_c import cache_path
    assert cache_path("team", "big", False, range(6), 1, smooth=13) != cache_path("team", "big", False, range(6), 1)


def test_default_smoothing_keeps_recipe_c_paths():
    import os
    from mtah.members_c import cache_path
    assert os.path.basename(cache_path("team", "big", False, tuple(range(6)), 1)) == "team_gruC_swap_met_k2_big_s012345_340546b95641c360.pt"
