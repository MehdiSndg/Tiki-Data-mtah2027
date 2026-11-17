import os
import sys

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")]
from mtah.members_c import ST_LABELLED, cache_path, st_tag  # noqa: E402


def test_default_key_unchanged():
    # the v30 members on disk must keep their file names
    p = cache_path("team", "big", False, tuple(range(6)), 1)
    assert os.path.basename(p) == "team_gruC_swap_met_k2_big_s012345_340546b95641c360.pt"


def test_st_tag_order_free_and_distinct():
    a, b = ST_LABELLED[:4], ST_LABELLED[4:]
    assert st_tag(a) == st_tag(tuple(reversed(a)))
    assert st_tag(a) != st_tag(b) != st_tag(ST_LABELLED)
    assert st_tag(()) == ""
    assert cache_path("team", "big", False, range(6), 1, st=a) != cache_path("team", "big", False, range(6), 1)


def test_pff_tag_distinct_and_default_unchanged():
    from mtah.members_c import pff_tag
    assert pff_tag(()) == ""
    a = cache_path("team", "tcn128", False, range(6), 1, pff=(3812, 3814))
    assert a != cache_path("team", "tcn128", False, range(6), 1) and "_pff" in a
    assert os.path.basename(cache_path("team", "big", False, tuple(range(6)), 1)) == "team_gruC_swap_met_k2_big_s012345_340546b95641c360.pt"
