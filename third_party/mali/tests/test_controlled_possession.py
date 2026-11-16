"""Gün 3: DFL touch-level labels -> Metrica-like controlled possession (labels.align)."""
from types import SimpleNamespace

import numpy as np

from mtah.constants import A, D, H
from mtah.labels.align import controlled_possession


def _scene(n=400):
    """Ball far from everyone; 2 home (slots 0, 1) + 2 away players (slots 2, 3)."""
    players = np.full((n, 4, 2), 50.0, np.float32)
    players[:, :, 1] = np.array([10.0, 20.0, 30.0, 40.0])
    ball = np.zeros((n, 2), np.float32)
    return SimpleNamespace(players=players, ball=ball, team=np.array([0, 0, 1, 1], np.int8))


def _labels(spec):
    return np.concatenate([np.full(k, c, np.int8) for c, k in spec])


def test_uncontrolled_short_return_is_merged_and_long_one_kept():
    lab = _labels([(H, 100), (A, 60), (H, 100), (A, 140), (H, 100)])
    out = controlled_possession(lab, [(0, len(lab))], _scene(len(lab)), max_frames=100, min_frames=19)
    assert (out[100:160] == H).all()            # 60-frame A touch without control -> H
    assert (out[260:400] == A).all()            # 140 frames > max_frames -> a real possession


def test_completed_pass_keeps_the_spell():
    lab = _labels([(H, 100), (A, 60), (H, 100)])
    sc = _scene(len(lab))
    sc.ball[100:160] = [50.0, 30.0]             # away player slot 2 at the ball ...
    sc.ball[130:160] = [50.0, 40.0]             # ... then slot 3: two different A players = a pass
    out = controlled_possession(lab, [(0, len(lab))], sc, max_frames=100, min_frames=19)
    assert (out[100:160] == A).all()


def test_below_min_frames_merged_even_with_control():
    lab = _labels([(H, 100), (A, 12), (H, 100)])
    sc = _scene(len(lab))
    sc.ball[100:106] = [50.0, 30.0]
    sc.ball[106:112] = [50.0, 40.0]
    out = controlled_possession(lab, [(0, len(lab))], sc, max_frames=100, min_frames=19)
    assert (out[100:112] == H).all()


def test_restart_and_touch_before_dead():
    lab = _labels([(D, 100), (A, 30), (H, 100), (A, 30), (D, 100)])
    sc = _scene(len(lab))
    keep = controlled_possession(lab, [(0, len(lab))], sc, before_dead=False)
    assert (keep[100:130] == A).all()           # D-B-A: B restarts the game = control
    assert (keep[230:260] == A).all()           # A-B-D kept without before_dead
    merged = controlled_possession(lab, [(0, len(lab))], sc, before_dead=True)
    assert (merged[230:260] == H).all() and (merged[100:130] == A).all()
    assert (merged == D).sum() == (lab == D).sum()


def test_dfl_target_gtn_merges_short_uncontrolled_returns():
    """Gün 3: --dfl-target gt20 = A-B-A returns <= 20 frames merged unless B shows control."""
    from types import SimpleNamespace as NS
    from mtah.pipeline.adapt import dfl_target
    lab = _labels([(H, 100), (A, 18), (H, 100), (A, 30), (H, 100)])
    sc = _scene(len(lab))
    data = NS(players=sc.players, ball=sc.ball, team=sc.team, labels=lab, period=np.ones(len(lab), np.int8))
    prep = {"data": data, "runs": [(0, len(lab))], "aligned": lab}
    out = dfl_target("gt20", prep, None, None)
    assert (out[100:118] == H).all() and (out[218:248] == A).all()
