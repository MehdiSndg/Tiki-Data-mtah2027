"""Gün 2: post-processing (snapping, short dead gaps), Smart Tagging N/A teams, broadcast fill."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mtah.constants import A, D, H, X
from mtah.data import TrackingData
from mtah.decoding import post
from mtah.features.groups.tags import resolve_na_teams


def test_short_dead_gap_closed_only_between_same_team_without_out_of_play():
    lab = np.array([H] * 50 + [D] * 40 + [H] * 50 + [D] * 40 + [A] * 50, np.int8)
    ball = np.full((len(lab), 2), 30.0, np.float32)
    out = post.close_short_dead(lab, [(0, len(lab))], ball, (105.0, 68.0), max_frames=60)
    assert np.all(out[:140] == H)                       # H D H -> H
    assert np.all(out[140:180] == D)                    # H D A stays
    ball[60] = (-2.0, 30.0)                             # ball out inside the first gap
    out = post.close_short_dead(lab, [(0, len(lab))], ball, (105.0, 68.0), max_frames=60)
    assert np.all(out[50:90] == D)


def test_snapping_moves_turnover_to_first_kick_and_dead_to_line_crossing():
    lab = np.array([H] * 100 + [A] * 100 + [D] * 100, np.int8)
    kick = np.zeros(300, bool)
    kick[92] = True                                     # first touch of A, 8 frames early
    oob = np.zeros(300, bool)
    oob[208] = True                                     # ball crosses the line 8 frames late
    out = post.snap_boundaries(lab, [(0, 300)], kick, oob, before=20, after=15)
    assert out[91] == H and out[92] == A
    assert out[207] == A and out[208] == D


def test_snapping_never_crosses_a_neighbour_boundary():
    lab = np.array([H] * 10 + [A] * 5 + [H] * 10, np.int8)
    kick = np.zeros(25, bool)
    kick[0] = True
    out = post.snap_boundaries(lab, [(0, 25)], kick, np.zeros(25, bool), before=20, after=15)
    starts = np.flatnonzero(np.diff(out) != 0) + 1
    assert len(starts) == 2 and out[0] == H             # segments survive, order kept


def test_oob_entries_only_on_visible_crossings():
    ball = np.array([[1, 30], [-1, 30], [-2, 30], [np.nan, np.nan], [-1, 30]], np.float32)
    ent = post.oob_entries(ball, (105.0, 68.0))
    assert ent.tolist() == [False, True, False, False, False]


def test_na_set_piece_teams_resolved_from_tracking_and_goals():
    n = 400
    players = np.full((n, 4, 2), np.nan, np.float32)
    players[:, 0] = (30, 40)
    players[:, 1] = (60, 0.2)                           # home player on the touchline
    players[:, 2] = (52.5, 34.2)                        # away player on the centre spot
    players[:, 3] = (70, 30)
    team = np.array([0, 0, 1, 1])
    # 8 bystanders so every frame has >= 10 players
    players = np.concatenate([players, np.tile(np.array([[50, 30]], np.float32), (n, 8, 1))], axis=1)
    team = np.r_[team, [0] * 4 + [1] * 4]
    data = SimpleNamespace(team=team)
    ctx = SimpleNamespace(length=105.0, width=68.0, P=players, data=data, n=n)
    t = pd.DataFrame([
        dict(code="GOALS", team=0, type="", start=10, end=50),
        dict(code="SET PIECES", team=-1, type="Kick Off (After Goal)", start=60, end=80),
        dict(code="SET PIECES", team=-1, type="Throw-in", start=100, end=200),
        dict(code="SET PIECES", team=-1, type="Kick Off (Start)", start=300, end=350),
    ])
    r = resolve_na_teams(t, ctx)
    assert r["team"].tolist() == [0, 1, 0, 1]


def test_broadcast_fill_only_fills_missing_frames(monkeypatch):
    from mtah.io import broadcast
    n = 10
    ball = np.full((n, 2), 5.0, np.float32)
    ball[3:6] = np.nan
    tac = TrackingData("barath", 25, (105.0, 68.0), np.arange(n), np.ones(n, np.int8),
                       np.zeros((n, 2, 2), np.float32), ball, np.array([0, 1]), None, {})
    fused = np.full((n, 2), 50.0, np.float32)
    fused[5] = np.nan
    monkeypatch.setattr(broadcast, "fused_ball", lambda t: fused)
    out = broadcast.fill_ball(tac)
    assert out.name == "barath_bf" and out.meta["broadcast_filled_frames"] == 2
    assert np.all(out.ball[:3] == 5.0) and np.all(out.ball[3:5] == 50.0) and np.isnan(out.ball[5, 0])
    assert np.isnan(tac.ball[3, 0])                     # the source is untouched


def test_alignment_merges_short_touch_but_keeps_a_controlled_spell():
    from mtah.labels.align import suppress_short_returns
    n = 200
    lab = np.array([A] * 60 + [H] * 20 + [A] * 60 + [H] * 20 + [A] * 40, np.int8)
    players = np.full((n, 3, 2), 50.0, np.float32)
    ball = np.full((n, 2), 10.0, np.float32)
    players[:, 2] = (11.0, 10.0)                         # an away player near the ball all along
    players[140:150, 0] = (10.2, 10.0)                   # second H spell: home player 0 touches ...
    players[150:160, 1] = (10.3, 10.0)                   # ... then home player 1: a completed pass
    data = SimpleNamespace(players=players, ball=ball, team=np.array([0, 0, 1]))
    out = suppress_short_returns(lab, [(0, n)], 35, data, check_control=True)
    assert np.all(out[60:80] == A)                       # 20-frame touch without control -> merged
    assert np.all(out[140:160] == H)                     # controlled (pass) -> kept
    assert np.all(suppress_short_returns(lab, [(0, n)], 11) == lab)   # old rule: both too long


def test_family_cfgs_expand_blend_members():
    from mtah.pipeline import engine
    cfg = {"model": {"families": ["lgbm", "catboost"]}, "lgbm": {}}
    fams = [c["model"]["family"] for c in engine.family_cfgs(cfg)]
    assert fams == ["lgbm", "catboost"]
    assert engine.family_cfgs({"lgbm": {}}) == [{"lgbm": {}}]
