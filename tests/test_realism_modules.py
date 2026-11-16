"""Characterisation tests for the 13 Nov realism modules (mtah.unglue, mtah.deadsim, mtah.pff controlled labels), written
after the code (each was checked to fail against a broken variant before it was kept)."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from test_degrade import toy_match  # noqa: E402


def glued_match(n=2000):
    m = toy_match(n)
    m["bx"] = m["px"][:, 3].copy()          # the ball sits on player 3 the whole time
    m["by"] = m["py"][:, 3].copy()
    return m


def nearest_m(m):
    L, W = m["pitch"]
    d = np.hypot((m["px"] - m["bx"][:, None]) * L, (m["py"] - m["by"][:, None]) * W)
    return np.nanmin(d, axis=1)


def test_unglue_moves_a_glued_ball_off_the_player():
    from mtah.unglue import unglue
    out = unglue(glued_match(), seed=0)
    assert np.median(nearest_m(out)) > 0.5


def test_unglue_leaves_a_free_ball_where_it_is():
    from mtah.unglue import unglue
    m = toy_match()
    far = nearest_m(m) > 1.0
    out = unglue(m, seed=0)
    assert np.array_equal(out["bx"][far], m["bx"][far])


def test_unglue_keeps_the_labels():
    from mtah.unglue import unglue
    m = glued_match()
    assert np.array_equal(unglue(m, seed=0)["labels"], m["labels"])


def dead_stretch_match(n=6000, a=2000, b=4000):
    m = toy_match(n)
    m["labels"][a:b] = pm.D
    return m, a, b


def test_realistic_dead_leaves_live_frames_alone():
    from mtah.deadsim import realistic_dead
    m, a, b = dead_stretch_match()
    out = realistic_dead(dict(m), m, seed=0)
    live = np.r_[np.arange(0, a), np.arange(b, len(m["bx"]))]
    assert np.array_equal(out["bx"][live], m["bx"][live])


def test_realistic_dead_shows_the_ball_on_part_of_the_dead_stretch():
    from mtah.deadsim import realistic_dead
    m, a, b = dead_stretch_match()
    seen = np.isfinite(realistic_dead(dict(m), m, seed=0)["bx"][a:b]).mean()
    assert 0.1 < seen < 0.8


def test_realistic_dead_shows_the_ball_more_often_near_the_restart():
    from mtah.deadsim import realistic_dead
    m, a, b = dead_stretch_match(n=12000, a=2000, b=10000)
    seen = np.mean([np.isfinite(realistic_dead(dict(m), m, seed=s)["bx"]) for s in range(200)], axis=0)
    assert seen[b - 300:b].mean() > 1.5 * seen[a + 100:a + 6000].mean()    # measured 0.246 vs 0.142 (flat profile: ~1.0)


def test_realistic_dead_ends_at_the_restart_spot():
    from mtah.deadsim import realistic_dead
    m, a, b = dead_stretch_match()
    for s in range(10):
        out = realistic_dead(dict(m), m, seed=s)
        if np.isfinite(out["bx"][b - 1]):
            assert abs(out["bx"][b - 1] - m["bx"][b]) * 105 < 1.5
            return
    raise AssertionError("the ball was never seen on the last dead frame in 10 draws")


def pff_events(kind_of_second):
    """Home passes at frame 10, away player makes a `kind_of_second` event at 60, home passes again at 110."""
    ev = []
    for f, home, pe in ((10, True, "PA"), (60, False, kind_of_second), (110, True, "PA")):
        ev.append({"gameEventType": "OTB", "start_frame": f, "end_frame": f + 5, "home_ball": home,
                   "possessionEventType": pe, "setpieceType": "O"})
    return ev


def test_controlled_labels_ignore_an_uncontrolled_touch_of_the_other_team():
    from mtah.pff import pff_labels
    frame, period = np.arange(200), np.ones(200, np.int8)
    lab = pff_labels(pff_events("CH"), frame, period, controlled=True)
    assert (lab[60:100] == pm.H).all()


def test_controlled_labels_follow_a_controlled_action_of_the_other_team():
    from mtah.pff import pff_labels
    frame, period = np.arange(200), np.ones(200, np.int8)
    lab = pff_labels(pff_events("IT"), frame, period, controlled=True)
    assert (lab[60:100] == pm.A).all()


def test_touch_labels_still_follow_any_touch():
    from mtah.pff import pff_labels
    frame, period = np.arange(200), np.ones(200, np.int8)
    lab = pff_labels(pff_events("CH"), frame, period)
    assert (lab[60:100] == pm.A).all()
