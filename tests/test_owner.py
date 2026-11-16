"""mtah.owner: the training-only target "which team has a player within 1.5 m of the ball in the clean tracking"
(the Barca-Hackathon fork's clean-owner head, P9eo)."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from test_degrade import toy_match  # noqa: E402


def on_player(m, j, frames):
    m["bx"][frames] = m["px"][frames, j]
    m["by"][frames] = m["py"][frames, j]


def far_from_everyone(m, frames):
    m["px"][frames] = 0.9
    m["py"][frames] = 0.9
    m["bx"][frames] = 0.1
    m["by"][frames] = 0.1


def test_ball_on_a_home_player_is_home_owned():
    from mtah.owner import owner_target
    m = toy_match(200)
    on_player(m, 3, slice(0, 50))                 # slot 3 is home (slot_team 0)
    y, w = owner_target(m)
    assert (y[:50, 0] == 1).all() and (w[:50] == 1).all()


def test_ball_on_an_away_player_is_away_owned():
    from mtah.owner import owner_target
    m = toy_match(200)
    on_player(m, 14, slice(0, 50))
    y, _ = owner_target(m)
    assert (y[:50, 1] == 1).all()


def test_ball_far_from_everyone_is_nobodys():
    from mtah.owner import owner_target
    m = toy_match(200)
    far_from_everyone(m, slice(0, 50))
    y, w = owner_target(m)
    assert (y[:50] == 0).all() and (w[:50] == 1).all()


def test_untracked_ball_and_ungraded_frames_carry_no_weight():
    from mtah.owner import owner_target
    m = toy_match(200)
    m["bx"][10:20] = np.nan
    m["half"][150:] = 0
    _, w = owner_target(m)
    assert (w[10:20] == 0).all() and (w[150:] == 0).all() and (w[20:150] == 1).all()


def test_swapping_the_teams_swaps_the_owner_columns():
    from mtah.match import swap_teams
    from mtah.owner import owner_target
    m = toy_match(300)
    on_player(m, 3, slice(0, 100))
    on_player(m, 14, slice(100, 200))
    y, w = owner_target(m)
    ys, ws = owner_target(swap_teams(m))
    assert np.array_equal(ys, y[:, ::-1]) and np.array_equal(ws, w)


def test_radius_is_in_metres():
    from mtah.owner import owner_target
    m = toy_match(10)
    m["px"][:] = 0.9
    m["py"][:] = 0.9
    m["px"][:, 3], m["py"][:, 3] = 0.5, 0.5
    m["bx"][:], m["by"][:] = 0.5 + 1.4 / 105, 0.5      # 1.4 m away along x
    assert (owner_target(m)[0][:, 0] == 1).all()
    m["bx"][:] = 0.5 + 1.6 / 105
    assert (owner_target(m)[0][:, 0] == 0).all()
