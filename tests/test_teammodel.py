import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.teammodel import decode  # noqa: E402


def test_decode_ignores_a_short_blip_and_keeps_a_long_change():
    p = np.r_[np.full(100, 0.8), np.full(5, 0.2), np.full(100, 0.8), np.full(100, 0.2)]
    alive = np.ones(len(p), bool)
    c = decode(p, alive, switch=8.0)
    assert (c[:205] == pm.H).all() and (c[215:] == pm.A).all()


def test_decode_writes_dead_outside_alive_runs():
    p = np.full(10, 0.9)
    alive = np.r_[np.ones(4, bool), np.zeros(2, bool), np.ones(4, bool)]
    c = decode(p, alive, switch=1.0)
    assert (c[4:6] == pm.D).all() and (c[alive] == pm.H).all()


def test_touch_features_carry_the_team_at_the_last_kink():
    from mtah.teammodel import touch_features
    n = 100
    # ball goes right at 10 m/s, turns back at frame 50 next to a home player (slot 0)
    bx = np.r_[np.linspace(0.30, 0.30 + 49 * 0.4 / 105, 50), np.linspace(0.30 + 49 * 0.4 / 105, 0.30 + 0.1 / 105, 50)]
    by = np.full(n, 0.5)
    px = np.full((n, 2), np.nan)
    py = np.full((n, 2), 0.5)
    px[:, 0] = bx[50]
    px[:, 1] = 0.9
    m = {"bx": bx, "by": by, "px": px, "py": py, "slot_team": np.array([0, 1]), "pitch": (105.0, 68.0)}
    f = touch_features(m)
    assert f["last_touch_team"][60] == 1 and f["last_touch_team"][40] == 0 and f["next_touch_team"][40] == 1


def test_motion_features_see_a_ball_flying_towards_a_home_player():
    from mtah.teammodel import motion_features
    n = 60
    # ball moves right at 15 m/s from x=20 m towards a home player at x=40 m; an away player behind it at x=10 m
    bx = (20 + 0.6 * np.arange(n)) / 105.0
    m = {"bx": bx, "by": np.full(n, 0.5), "px": np.c_[np.full(n, 40 / 105.0), np.full(n, 10 / 105.0)],
         "py": np.full((n, 2), 0.5), "slot_team": np.array([0, 1]), "pitch": (105.0, 68.0)}
    f = motion_features(m)
    assert f["ball_to_team"][20] > 0.5          # + = heading to a home player


def test_reach_features_give_the_loose_ball_to_the_team_that_gets_there_first():
    from mtah.teammodel import reach_features
    n = 40
    # ball rolls right at 8 m/s from x=30 m; a home player 5 m ahead of it on its path, an away player 20 m behind
    bx = (30 + 0.32 * np.arange(n)) / 105.0
    m = {"bx": bx, "by": np.full(n, 0.5), "px": np.c_[np.full(n, 45 / 105.0), np.full(n, 10 / 105.0)],
         "py": np.full((n, 2), 0.5), "slot_team": np.array([0, 1]), "pitch": (105.0, 68.0)}
    f = reach_features(m)
    assert f["reach_team"][10] == 1 and f["reach_margin"][10] > 0


def test_decode3_puts_a_short_dead_spell_where_the_dead_probability_says():
    from mtah.teammodel import decode3
    n = 300
    p_home = np.r_[np.full(150, 0.9), np.full(150, 0.1)]
    p_dead = np.zeros(n)
    p_dead[140:170] = 0.95
    c = decode3(p_home, p_dead, np.ones(n, bool), switch=12.0, dead_cost=4.0)
    assert (c[145:165] == pm.D).all() and (c[:130] == pm.H).all() and (c[180:] == pm.A).all()


def test_carry_features_see_an_away_player_dribbling():
    from mtah.teammodel import carry_features
    n = 50
    x = (30 + 0.25 * np.arange(n)) / 105.0          # ball and away player move together at 6.25 m/s
    m = {"bx": x, "by": np.full(n, 0.5), "px": np.c_[np.full(n, 20 / 105.0), x - 0.8 / 105.0],
         "py": np.full((n, 2), 0.5), "slot_team": np.array([0, 1]), "pitch": (105.0, 68.0)}
    f = carry_features(m)
    assert f["carry_team"][25] == -1


def test_decode_varying_switch_lets_a_short_spell_through_where_the_cost_is_low():
    from mtah.teammodel import decode
    p = np.r_[np.full(100, 0.8), np.full(8, 0.2), np.full(100, 0.8)]
    alive = np.ones(len(p), bool)
    flat = decode(p, alive, 12.0)
    cost = np.full(len(p), 12.0)
    cost[95:115] = 1.0
    low = decode(p, alive, cost)
    assert (flat == pm.H).all() and (low[101:107] == pm.A).all()


def _toy_match(n=300):
    m = {"pitch": (105.0, 68.0), "px": np.full((n, 2), 0.5), "py": np.full((n, 2), 0.5), "slot_team": np.array([0, 1]),
         "gk": {0: 0, 1: 1}, "bx": np.full(n, 0.5), "by": np.full(n, 0.5), "half": np.ones(n, np.int8), "sign": {1: 1}}
    m["px"][:, 0], m["px"][:, 1] = 0.05, 0.95            # home keeper near x=5 m, away keeper near x=100 m
    return m


def test_keeper_features_flag_a_ball_lost_next_to_a_keeper():
    from mtah.teammodel import keeper_features
    m = _toy_match()
    m["bx"][:100] = np.linspace(0.5, 0.06, 100)          # ball travels to the home keeper ...
    m["bx"][100:] = np.nan                                # ... and is lost there
    f = keeper_features(m)
    assert f["lost_at_gk"][150] == 1 and f["lost_at_gk"][50] == 0
    assert f["last_d_gkH"][150] < 3 and f["last_d_gkA"][150] > 80


def test_episode_features_count_touch_runs_and_changes():
    from mtah.teammodel import episode_features
    t = np.zeros(300)
    t[[10, 20, 30]] = 1                                   # three home touches
    t[[40, 50]] = -1                                      # then two away touches
    f = episode_features(t)
    assert f["touch_run"][35] == 3 and f["touch_run"][55] == -2
    assert f["touch_changes_r75"][60] >= 1 and f["touch_changes_r75"][5] == 0


def test_vel_backward_matches_the_old_formula_and_centered_has_no_lag():
    import numpy as np
    from mtah import teammodel as T
    x = np.r_[np.zeros(50), np.arange(1, 51, dtype=float)]          # still, then 1 m per frame from frame 50
    v = T._vel(x, 5)
    assert np.isnan(v[:5]).all() and np.allclose(v[5:], (x[5:] - x[:-5]) * 25 / 5)
    c = T._vel(x, 5, centered=True)
    assert np.isnan(c[:2]).all() and np.isnan(c[-2:]).all()
    assert np.isclose(c[60], 25.0) and np.isclose(c[40], 0.0)
    assert np.argmax(c > 12) < np.argmax(v > 12)                    # the centred speed rises earlier
    X = np.c_[x, 2 * x]
    assert np.allclose(T._vel(X, 5)[10:], np.c_[v, 2 * v][10:])
