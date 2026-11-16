import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.external import soccertrack_labels, soccertrack_side_codes  # noqa: E402


def test_labels_map_codes_and_carry_loose_ball():
    status = np.array([0, 1, 1, 3, 3, 2, 3, 0, 3, 2], np.int8)
    c = soccertrack_labels(status, home_code=1)
    assert c.tolist() == [pm.D, pm.H, pm.H, pm.H, pm.H, pm.A, pm.A, pm.D, pm.A, pm.A]


def test_labels_other_home_code():
    c = soccertrack_labels(np.array([2, 1, 0], np.int8), home_code=2)
    assert c.tolist() == [pm.H, pm.A, pm.D]


def test_labels_unknown_status_keeps_previous():
    c = soccertrack_labels(np.array([1, -1, 1], np.int8), home_code=1)
    assert c.tolist() == [pm.H, pm.H, pm.H]


def test_side_codes_by_majority_of_team_actions():
    status = np.array([1] * 10 + [2] * 10, np.int8)
    events = [{"label": "PASS", "frame": f, "team": "left"} for f in (2, 3, 4, 15)] + \
             [{"label": "DRIVE", "frame": f, "team": "right"} for f in (12, 13, 14)] + \
             [{"label": "OUT", "frame": 5, "team": "right"}]
    assert soccertrack_side_codes(events, status) == {"left": 1, "right": 2}
