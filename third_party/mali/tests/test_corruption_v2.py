"""Gün 6 (Ek P.7): ATD-like DFL corruption v2 -- dead-ball carry, player smoothing and dropouts."""
import numpy as np

from mtah import constants as C
from mtah.pipeline import dfl_referee


def _toy(n=3000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    players = np.cumsum(rng.normal(0, 0.2, (n, k, 2)), axis=0).astype(np.float32) + 50
    ball = players[:, 0, :] + 0.5
    labels = np.full(n, C.H, np.int8)
    labels[1000:1400] = C.D
    labels[2000:2300] = C.D
    period = np.ones(n, np.int8)
    return players, ball.astype(np.float32), labels, period


def test_defaults_are_off():
    for k in ("dead_carry_prob", "player_smooth_frames", "player_gap_frac"):
        assert dfl_referee.DEFAULTS[k] == 0


def test_dead_carry_follows_nearest_player_and_keeps_nan():
    players, ball, labels, period = _toy()
    ball[1100:1150] = np.nan
    out = dfl_referee.carry_dead_ball(ball, players, labels, period, 1.0, 0.3, np.random.default_rng(1))
    assert np.isnan(out[1100:1150]).all()
    d = np.linalg.norm(out[1000:1100] - players[1000:1100, 0], axis=1)
    assert np.allclose(d, 0.3, atol=1e-4)                       # follows player 0 at a fixed offset
    assert np.array_equal(out[:1000], ball[:1000])              # live frames untouched
    off = dfl_referee.carry_dead_ball(ball, players, labels, period, 0.0, 0.3, np.random.default_rng(1))
    assert np.array_equal(off, ball, equal_nan=True)


def test_player_dropouts_hit_the_target_share_and_are_deterministic():
    players, *_ = _toy(n=60000, k=6)
    a = dfl_referee.atd_like_players(players, 0, 0.07, 3.8, 25, np.random.default_rng(3))
    b = dfl_referee.atd_like_players(players, 0, 0.07, 3.8, 25, np.random.default_rng(3))
    assert np.array_equal(a, b, equal_nan=True)
    share = np.isnan(a[..., 0]).mean()
    assert 0.04 < share < 0.10, share
    assert np.array_equal(a[~np.isnan(a)], players[~np.isnan(a)])   # no smoothing requested


def test_player_smoothing_reduces_frame_to_frame_speed_change():
    players, *_ = _toy()
    s = dfl_referee.atd_like_players(players, 15, 0.0, 3.8, 25, np.random.default_rng(0))
    dv = lambda p: np.median(np.linalg.norm(np.diff(np.diff(p, axis=0), axis=0), axis=-1))  # noqa: E731
    assert dv(s) < 0.3 * dv(players)


def test_eval_cfg_overrides_only_the_referee_and_is_absent_by_default():
    from mtah import config
    from mtah.validation.splits import cache_tag
    cfg = config.load("main_v1")
    assert dfl_referee.eval_cfg(cfg) is None and dfl_referee.eval_tag(cfg) == ""
    ev = dict(cfg, dfl_referee_eval={"player_smooth_frames": 15, "dead_carry_prob": 0.25})
    e = dfl_referee.eval_cfg(ev)
    assert e["dfl_referee"]["player_smooth_frames"] == 15 and e["dfl_referee"]["dead_carry_prob"] == 0.25
    assert cfg.get("dfl_referee", {}).get("player_smooth_frames", 0) == 0   # original untouched
    assert dfl_referee.eval_tag(ev).startswith("_ev")
    assert cache_tag(ev) == cache_tag(cfg)          # training caches are shared with the plain config


def test_state_gaps_and_ghosts_hit_targets():
    rng = np.random.default_rng(0)
    n, k = 60000, 40
    players = np.full((n, k, 2), np.nan, np.float32)
    players[:, :11] = rng.normal(50, 10, (n, 11, 2))
    players[:, 20:31] = rng.normal(50, 10, (n, 11, 2))
    team = np.r_[np.zeros(20, np.int8), np.ones(20, np.int8)]
    labels = np.full(n, C.H, np.int8)
    labels[30000:45000] = C.D
    out = dfl_referee.state_gaps_and_ghosts(players, team, labels, 25, 0.02, 1.0, 0.13, 7.0, 0.06, 3.0, 2.0,
                                            np.random.default_rng(1))
    vis = ~np.isnan(out[..., 0])
    live, dead = labels != C.D, labels == C.D
    home_live = vis[live][:, :20].sum(1)
    home_dead = vis[dead][:, :20].sum(1)
    assert 10.6 < home_live.mean() < 11.1 and 9.2 < home_dead.mean() < 10.0
    assert 0.02 < np.mean(home_live > 11) < 0.10                 # ghost tracks exist
    again = dfl_referee.state_gaps_and_ghosts(players, team, labels, 25, 0.02, 1.0, 0.13, 7.0, 0.06, 3.0, 2.0,
                                              np.random.default_rng(1))
    assert np.array_equal(out, again, equal_nan=True)


def test_mixed_training_flag_changes_cache_tag_only_when_set():
    from mtah import config
    from mtah.validation.splits import cache_tag
    cfg = config.load("main_v1")
    ev = dict(cfg, dfl_referee_eval={"player_smooth_frames": 15})
    mix = dict(ev, dfl_mix_eval=True)
    assert cache_tag(ev) == cache_tag(cfg) != cache_tag(mix)
