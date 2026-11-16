import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.setgru import NFP, player_set, set_net  # noqa: E402


def toy(n=10):
    px = np.full((n, 4), np.nan, np.float32)
    py = np.full((n, 4), 0.5, np.float32)
    px[:, 0], px[:, 1], px[:, 2] = 0.52, 0.60, 0.45          # home slots 0, 1; away slot 2; slot 3 absent
    bx = np.full(n, 0.5, np.float32)
    bx[3:6] = np.nan                                          # ball lost: last seen is used
    return {"px": px, "py": py, "bx": bx, "by": np.full(n, 0.5, np.float32), "pitch": (100.0, 50.0),
            "slot_team": np.array([0, 0, 1, 1], np.int8), "half": np.ones(n, np.int8), "sign": {1: -1, 2: 1}}


def test_player_set_nearest_first_and_presence():
    P = player_set(toy(), k=2).reshape(10, 2, 2, NFP)
    assert np.allclose(P[4, 0, :, 2], [2.0, 10.0])           # home, by distance, ball from the last sighting
    assert P[4, 0, :, -1].tolist() == [1, 1]
    assert P[4, 1, :, -1].tolist() == [1, 0] and np.isnan(P[4, 1, 1, 2])
    assert np.isclose(P[4, 0, 0, 0], -2.0)                   # +2 m in x, home attacking towards -x


def test_set_net_shapes():
    import torch
    net = set_net(3, k=2)
    y = net(torch.zeros(4, 7, 3 + 2 * 2 * NFP))
    assert tuple(y.shape) == (4, 7)
