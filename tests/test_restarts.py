import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.restarts import FPS, HOLD, W, restart_evidence  # noqa: E402


def ev(start, end, team="Barcelona", typ="Corner Kick"):
    return {"code": "SET PIECES", "start_frame": start, "end_frame": end, "tags": {"Team": team, "Type": typ}}


def test_default_fires_on_the_first_alive_frame_in_the_window():
    alive = np.ones(1000, bool)
    alive[300:400] = False
    E = restart_evidence([ev(100, 500)], alive)
    assert E[100] == W                                   # old behaviour: the set piece's start, still in play


def test_strict_fires_only_where_a_dead_block_ends():
    alive = np.ones(1000, bool)
    alive[300:400] = False
    E = restart_evidence([ev(100, 500)], alive, strict=True)
    assert E[100] == 0 and E[400] == W and E[399] == 0 and E[400 + HOLD - 1] == W and E[400 + HOLD] == 0


def test_strict_skips_a_set_piece_without_a_dead_block():
    alive = np.ones(1000, bool)
    assert not restart_evidence([ev(100, 500)], alive, strict=True).any()


def test_strict_accepts_a_window_that_starts_inside_the_dead_block():
    alive = np.ones(1000, bool)
    alive[50:150] = False
    E = restart_evidence([ev(100, 500, team="Atletic Club")], alive, strict=True)
    assert E[150] == -W and not E[100:150].any()
