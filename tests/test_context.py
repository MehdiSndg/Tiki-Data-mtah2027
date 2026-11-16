import numpy as np

from conftest import needs_data
from mtah.context import SETPIECE_TYPES, tag_features, whistle_feature


def ev_(code, team, start, end, **tags):
    return {"code": code, "start_frame": start, "end_frame": end, "tags": dict(Team=team, **tags)}


def test_mirrored_pair_counted_once():
    ev = [ev_("BUILD UP", "Barcelona", 0, 50), ev_("DEFENDING IN ATTACKING THIRD", "Atletic Club", 0, 50)]
    f = tag_features(ev, 100)
    assert f["tag_H"][10] == 1 and f["tag_A"][10] == 0 and f["tag_H"][60] == 0


def test_unmirrored_defending_tag_still_counts_for_the_other_side():
    f = tag_features([ev_("DEFENDING IN MIDDLE THIRD", "Barcelona", 0, 50)], 100)
    assert f["tag_A"][10] == 1 and f["tag_H"][10] == 0


def test_na_team_ignored_for_side_but_kept_for_setpiece():
    f = tag_features([ev_("SET PIECES", "N/A", 20, 30, Type="Throw-in")], 100)
    assert f["tag_H"].sum() == f["tag_A"].sum() == 0 and f["setpiece"][25] and not f["setpiece"][30]
    assert f["setpiece_type"][25] == SETPIECE_TYPES.index("throw_in")


@needs_data
def test_whistle_feature_length_and_anchor():
    w = whistle_feature(199319)
    assert w.shape == (199319,) and w[11185 - 75:11185 + 75].max() > 1000   # 1st-half kickoff whistle (EDA F)
