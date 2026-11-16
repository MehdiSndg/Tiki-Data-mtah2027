import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation"))
import possession_metrics as pm  # noqa: E402

PITCH = (100.0, 50.0)


def _match(bx, by, px, py, slot_team):
    return {"bx": np.asarray(bx, np.float32), "by": np.asarray(by, np.float32),
            "px": np.asarray(px, np.float32), "py": np.asarray(py, np.float32),
            "slot_team": np.asarray(slot_team, np.int8), "pitch": PITCH}


def test_transitions_skip_dead_breaks():
    from mtah.timing import transitions
    H, A, D, X = pm.H, pm.A, pm.D, pm.X
    c = np.array([H, H, A, A, D, A, A, D, H, X, A, H], np.int8)
    f, new = transitions(c)
    assert f.tolist() == [2, 11] and new.tolist() == [A, H]     # H->D->A and X breaks are not hand-overs


def test_contact_lag_signed():
    from mtah.timing import contact_lag
    n = 100
    bx, by = np.full(n, 0.5), np.full(n, 0.5)
    px = np.full((n, 2), 0.9)
    py = np.full((n, 2), 0.5)
    px[40:, 1] = 0.5 + 1.0 / PITCH[0]                # away player 1 m from the ball from frame 40 on
    m = _match(bx, by, px, py, [0, 1])
    assert abs(contact_lag(m, 50, pm.A) - 10 / 25) < 1e-9     # label 10 frames after contact
    assert abs(contact_lag(m, 35, pm.A) + 5 / 25) < 1e-9      # label 5 frames before contact
    assert np.isnan(contact_lag(m, 50, pm.H))                 # home never near


def test_contact_lag_nan_without_ball():
    from mtah.timing import contact_lag
    n = 100
    px = np.full((n, 1), 0.5)
    m = _match(np.full(n, np.nan), np.full(n, np.nan), px, px.copy(), [1])
    assert np.isnan(contact_lag(m, 50, pm.A))


def test_still_runs_in_metres():
    from mtah.timing import still_runs
    n = 200
    x = np.full(n, 0.5)
    x[100:] = 0.5 + np.arange(100) * (1.0 / 25) / PITCH[0]      # 1 m/s from frame 100: not still
    runs = still_runs(x, np.full(n, 0.5), PITCH, v_max=0.5, min_s=1.0)
    assert len(runs) == 1 and runs[0][0] == 0 and 99 <= runs[0][1] <= 101
    x2 = x.copy()
    x2[30:35] = np.nan                                          # a gap ends a run; 30 frames < 1 s
    assert all(b - a >= 25 for a, b in still_runs(x2, np.full(n, 0.5), PITCH, v_max=0.5, min_s=1.0))
    assert not any(a < 30 <= b - 1 for a, b in still_runs(x2, np.full(n, 0.5), PITCH))


def test_kinks_find_a_turn():
    from mtah.timing import kinks
    n = 50
    t = np.arange(n)
    x = np.where(t < 25, t * 0.4, 25 * 0.4) / PITCH[0]          # 10 m/s along x ...
    y = np.where(t < 25, 0.0, (t - 25) * 0.4) / PITCH[1] + 0.5  # ... then 10 m/s along y
    k = kinks(x, y, PITCH)
    assert len(k) >= 1 and np.all(np.abs(k - 25) <= 2)
    assert len(kinks(t * 0.4 / PITCH[0], np.full(n, 0.5), PITCH)) == 0     # a straight line has none
