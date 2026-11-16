import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.deadctx import CAP_S, FPS, dead_context, restart_team  # noqa: E402


def toy(n=1000):
    px = np.full((n, 2), np.nan, np.float32)
    py = np.full((n, 2), 0.5, np.float32)
    px[:, 0], px[:, 1] = 0.30, 0.70                 # home slot 0 at x 0.3, away slot 1 at x 0.7
    bx, by = np.full(n, 0.69, np.float32), np.full(n, 0.5, np.float32)
    bx[400:500] = np.nan                            # dead stretch, ball lost
    bx[500:] = 0.31                                 # restart next to the home player
    return {"px": px, "py": py, "bx": bx, "by": by, "pitch": (100.0, 50.0), "slot_team": np.array([0, 1], np.int8),
            "half": np.ones(n, np.int8)}


def test_restart_team_is_nearest_team_at_first_sighting_after_the_block():
    assert restart_team(toy(), 500) == 1           # home
    m = toy()
    m["bx"][500:] = 0.69
    assert restart_team(m, 500) == -1              # away


def test_dead_context_times_and_teams():
    m = toy()
    dead = np.zeros(1000, bool)
    dead[400:500] = True
    C = dead_context(m, dead)
    assert C.shape == (1000, 5)
    assert np.isclose(C[300, 0], 100 / FPS)        # until the next dead block
    assert np.isclose(C[300, 2], 1)                # it is restarted by home
    assert np.isclose(C[600, 1], 100 / FPS)        # since the last dead block
    assert np.isclose(C[600, 3], 1)                # which home restarted
    assert np.isclose(C[600, 0], CAP_S) and C[600, 2] == 0     # no next block
    assert np.allclose(C[450], 0)                  # dead frames carry nothing
    assert np.isclose(C[300, 4], 25.0)             # last sighting before the next block: 25 m from a touch line (31 m from a goal line)
