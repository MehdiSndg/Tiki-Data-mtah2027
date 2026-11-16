"""Gün 5 (Ek N.13): whistle timing (decoding.whistle), off by default."""
import numpy as np

from mtah.constants import A, D, H
from mtah.decoding import whistle


def test_off_by_default():
    assert whistle.params({"decoder": {}})["enabled"] is False


def test_moves_late_dead_start_to_the_deceleration_and_keeps_segments():
    n = 1000
    p80 = np.where(np.arange(n) < 400, 3.5, 1.5)          # collective speed collapses at frame 400
    lab = np.full(n, H, np.int8)
    lab[450:800] = D                                        # decoder put the stoppage 50 frames late
    lab[800:] = A
    wp = dict(whistle.DEFAULTS, enabled=True, min_drop=1.0, min_move=10, shift=0)
    out = whistle.refine_dead_starts(lab, [(0, n)], p80, wp, info := {})
    start = int(np.flatnonzero(out == D)[0])
    assert abs(start - 400) <= 8 and info["n_whistle_moved"] == 1
    assert (out[800:] == A).all() and np.count_nonzero(np.diff(out)) == np.count_nonzero(np.diff(lab))


def test_small_or_weak_moves_are_ignored():
    n = 1000
    p80 = np.where(np.arange(n) < 445, 3.5, 3.2)          # weak drop
    lab = np.full(n, H, np.int8)
    lab[450:800] = D
    out = whistle.refine_dead_starts(lab, [(0, n)], p80, dict(whistle.DEFAULTS, enabled=True, min_drop=1.0))
    assert (out == lab).all()
