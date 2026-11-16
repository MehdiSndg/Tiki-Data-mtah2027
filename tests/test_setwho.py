import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.degrade import pm  # noqa: E402
from mtah.setgru import NFP  # noqa: E402
from mtah.setwho import possessor_target, set_net_who, slot_columns  # noqa: E402
from test_setgru import toy  # noqa: E402


def test_slot_columns_follow_player_set_order():
    cols = slot_columns(toy(), k=2)
    assert cols.shape == (10, 2, 2)
    assert cols[4, 0].tolist() == [0, 1] and cols[4, 1].tolist() == [2, -1]


def test_possessor_target_marks_labelled_teams_player_nearest_true_ball():
    m = toy()
    m["labels"] = np.full(10, pm.H, np.int8)
    m["labels"][8:] = pm.A
    tbx, tby = np.full(10, 0.6, np.float32), np.full(10, 0.5, np.float32)     # true ball on home slot 1
    y = possessor_target(m, tbx, tby, k=2)
    assert y.shape == (10, 4)
    assert y[4].tolist() == [0, 1, 0, 0]        # home block, rank 1 (second nearest the last seen ball)
    assert y[9].tolist() == [0, 0, 0, 0]        # labelled away: away slot 2 is 15 m from the true ball


def test_set_net_who_shape():
    import torch
    net = set_net_who(3, k=2)
    y = net(torch.zeros(4, 7, 3 + 2 * 2 * NFP))
    assert tuple(y.shape) == (4, 7, 5)
