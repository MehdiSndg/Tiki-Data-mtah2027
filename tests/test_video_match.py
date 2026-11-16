import numpy as np
import pytest

from conftest import needs_data


@pytest.fixture(scope="module")
def both():
    from mtah.match import from_tactical
    from mtah.video_match import from_tactical_v2
    return from_tactical(), from_tactical_v2()


@needs_data
def test_v2_keeps_atd_where_it_is_seen(both):
    v1, v2 = both
    seen = ~np.isnan(v1["bx"])
    assert np.array_equal(v1["bx"][seen], v2["bx"][seen]) and np.array_equal(v1["by"][seen], v2["by"][seen])
    assert (~np.isnan(v2["bx"])).sum() > seen.sum()            # and fills some of the gaps


@needs_data
def test_v2_keeps_verified_keepers_and_drops_ghosts(both):
    from mtah.data import GK_SLOTS
    from mtah.frames import load_frames
    from mtah.keepers import GHOST, SEEN
    v1, v2 = both
    df = load_frames()
    for team, side in ((0, "H"), (1, "A")):
        k, st = GK_SLOTS[team], df[side + "_gk_status"].values
        assert np.array_equal(v1["px"][st == SEEN, k], v2["px"][st == SEEN, k])
        ghost = st == GHOST
        moved = ~np.isnan(v2["px"][ghost, k])
        assert np.all(v2["px"][ghost, k][moved] != v1["px"][ghost, k][moved])   # replaced, never kept
    assert v2["name"] == "tactical_v2" and v2["public_gt"]


def test_keeper_tracklets_drop_isolated_jumps():
    from mtah.video_match import keeper_tracklets
    f = np.arange(30)
    x = np.full(30, 10.0)
    y = np.full(30, 30.0)
    x[12] = 40.0                                     # one frame on someone else, 30 m away
    keep = keeper_tracklets(f, x, y)
    assert not keep[12] and keep[:12].all() and keep[13:].all()
    f2 = np.r_[np.arange(10), np.arange(100, 103)]   # a 3-frame tracklet on its own
    keep2 = keeper_tracklets(f2, np.full(13, 10.0), np.full(13, 30.0))
    assert keep2[:10].all() and not keep2[10:].any()


@needs_data
def test_v2_leaves_the_broadcast_ball_out_by_default(both):
    from mtah.balltrack import BROADCAST
    _, v2 = both
    assert not np.any(v2["ball_source"] == BROADCAST)       # it hurt the team call (eda_out/reports/P3.md)


@needs_data
def test_v2_takes_a_given_detector_track(both):
    from mtah.video_match import from_tactical_v2
    v1, _ = both
    n = len(v1["bx"])
    none = (np.full((n, 2), np.nan), np.zeros(n), np.zeros(n, bool))
    v2 = from_tactical_v2(det=none)                  # no detector boxes: only interpolation can fill
    assert np.array_equal(v2["bx"][~np.isnan(v1["bx"])], v1["bx"][~np.isnan(v1["bx"])])
    from mtah.balltrack import DETECTOR
    assert not np.any(v2["ball_source"] == DETECTOR)
