import os
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from test_degrade import toy_match

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.harness import evaluate  # noqa: E402
from mtah.labelers import baseline_smooth, control, nearest, own_dead  # noqa: E402
from mtah.match import from_tactical  # noqa: E402


@needs_data
def test_harness_reproduces_the_official_baseline_score():
    r = evaluate(lambda m: baseline_smooth(m, 50), [from_tactical()], bootstrap=0)
    assert r.loc[0, "spell_f1"] == pytest.approx(0.3256, abs=1e-4)   # README.md scoreboard


@needs_data
def test_baseline_copy_is_frame_identical_to_the_published_file():
    official = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")).label.values
    ours = np.array(pm.LABELS)[baseline_smooth(from_tactical(), 50)]
    assert np.array_equal(ours, official)


def test_nearest_gives_the_team_of_the_closest_player_and_d_without_ball():
    m = toy_match(n=10)
    m["px"][:], m["py"][:] = np.nan, np.nan
    m["px"][:, 3], m["py"][:, 3] = 0.50, 0.50                 # home
    m["px"][:, 14], m["py"][:, 14] = 0.60, 0.50               # away
    m["bx"][:] = np.r_[[0.51] * 5, [0.59] * 4, [np.nan]]
    m["by"][:] = 0.5
    assert nearest(m).tolist() == [pm.H] * 5 + [pm.A] * 4 + [pm.D]


def test_control_needs_time_on_the_ball_and_respects_dead_frames():
    m = toy_match(n=200)
    m["px"][:], m["py"][:] = np.nan, np.nan
    m["px"][:, 3], m["py"][:, 3] = 0.30, 0.50                 # home, far from the ball after frame 50
    m["px"][:, 14], m["py"][:, 14] = 0.70, 0.50               # away
    m["bx"][:] = np.r_[np.full(50, 0.30), np.full(5, 0.70), np.full(5, 0.5), np.full(140, 0.70)]
    m["by"][:] = 0.5
    m["labels"][:] = pm.H
    m["labels"][180:] = pm.D
    out = control(m, R=1.5, T=0.4, dead="truth")
    assert np.all(out[:50] == pm.H)
    assert np.all(out[50:69] == pm.H)                         # 5 frames at the away player: too short
    assert np.all(out[69:180] == pm.A) and np.all(out[180:] == pm.D)   # 10 frames (0.4 s) from frame 60


def test_own_dead_needs_a_missing_ball_and_slow_players():
    m = toy_match(n=500)
    m["px"][:] = 0.5 + 0.0016 * np.arange(500)[:, None] * np.r_[1, -1][np.arange(22) % 2]   # 4.2 m/s
    m["px"][200:400] = m["px"][200]                           # everyone stops for 8 s
    m["bx"][250:350] = np.nan                                 # ball gone inside the stop
    m["bx"][50:100] = np.nan                                  # ball gone while players run
    dead = own_dead(m)
    assert dead[260:340].all() and not dead[60:90].any() and not dead[:40].any()


def test_evaluate_returns_one_row_per_match():
    m = toy_match(n=500)
    m["labels"][250:] = pm.A
    m["labels"][480:] = pm.D
    r = evaluate(lambda x: x["labels"], [m, dict(m, name="toy2")], bootstrap=0)
    assert list(r["name"]) == ["toy", "toy2"]
    assert np.allclose(r["spell_f1"], 1.0) and np.allclose(r["macro_f1"], 1.0)
    assert {"n_pred", "n_gt", "boot_low", "boot_high"} <= set(r.columns)


@needs_data
def test_control_refuses_to_take_the_dead_ball_from_the_public_ground_truth():
    with pytest.raises(ValueError):
        control(from_tactical(), dead="truth")
