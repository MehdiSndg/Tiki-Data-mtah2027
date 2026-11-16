import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.external import SKILLCORNER, fill_group, from_skillcorner, skillcorner_labels, to_25fps  # noqa: E402

HOME, AWAY = 7, 9


def phases(rows):
    return pd.DataFrame(rows, columns=["frame_start", "frame_end", "team_in_possession_id"])


def test_labels_phase_team_and_dead_between():
    period = np.r_[np.zeros(2), np.ones(100)].astype(np.int8)
    c = skillcorner_labels(phases([(2, 30, HOME), (30, 50, AWAY), (90, 102, HOME)]), period, HOME)
    assert (c[:2] == pm.X).all()
    assert (c[2:30] == pm.H).all() and (c[30:50] == pm.A).all()
    assert (c[50:90] == pm.D).all()                  # a 4 s gap is a stoppage
    assert (c[90:] == pm.H).all()


def test_labels_short_gap_keeps_previous_team():
    period = np.ones(100, np.int8)
    c = skillcorner_labels(phases([(0, 30, AWAY), (40, 100, HOME)]), period, HOME)   # 1 s gap
    assert (c[30:40] == pm.A).all()


def test_labels_short_gap_not_filled_across_periods():
    period = np.r_[np.ones(50), np.full(50, 2)].astype(np.int8)
    c = skillcorner_labels(phases([(0, 45, AWAY), (55, 100, HOME)]), period, HOME)
    assert (c[45:55] == pm.D).all()


def test_fill_group_follows_possessor_team_and_carries_it():
    ph = np.array([pm.X, pm.H, pm.H, pm.H, pm.H, pm.D, pm.A, pm.A], np.int8)
    grp = np.array([-1, -1, pm.A, -1, pm.H, pm.A, -1, -1], np.int8)
    assert fill_group(ph, grp).tolist() == [pm.X, pm.H, pm.A, pm.A, pm.H, pm.D, pm.A, pm.A]


def test_fill_group_does_not_carry_across_a_stoppage():
    ph = np.array([pm.H, pm.H, pm.D, pm.H, pm.H], np.int8)
    grp = np.array([pm.A, -1, -1, -1, -1], np.int8)
    assert fill_group(ph, grp).tolist() == [pm.A, pm.A, pm.D, pm.H, pm.H]


def test_to_25fps_interpolates_and_keeps_nan():
    a = np.array([0.0, 1.0, 2.0, np.nan, 4.0], np.float32)
    b = to_25fps(a, 11)
    assert b[0] == 0.0 and abs(b[1] - 0.4) < 1e-6 and abs(b[2] - 0.8) < 1e-6
    assert b[5] == 2.0 and b[10] == 4.0               # exactly on samples 2 and 4 (t = 0.2, 0.4 s)
    assert np.isnan(b[6]) and np.isnan(b[9])          # a neighbour is the NaN sample


def test_to_25fps_codes_take_previous_sample():
    c = np.array([pm.H, pm.A, pm.D], np.int8)
    assert to_25fps(c, 7).tolist() == [pm.H, pm.H, pm.H, pm.A, pm.A, pm.D, pm.D]


@pytest.mark.skipif(not os.path.isdir(SKILLCORNER), reason="SkillCorner open data not downloaded")
def test_from_skillcorner_real_match():
    m = from_skillcorner("1874553")
    n = len(m["labels"])
    assert m["px"].shape[0] == n == len(m["bx"]) == len(m["half"])
    g = m["half"] > 0
    assert 0.40 < np.mean(m["labels"][g] == pm.D) < 0.55
    assert 0.40 < np.mean(~np.isnan(m["bx"][g])) < 0.75
    spells = pm.possession_spells(m["labels"], pm.valid_runs(m["labels"]))
    assert 250 < len(spells) < 450
    assert m["sign"][1] == -m["sign"][2]
