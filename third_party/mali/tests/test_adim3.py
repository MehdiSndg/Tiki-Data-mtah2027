"""Adım 3: state-aware masking, dead-ball labelling functions, band floor, source weights."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mtah.constants import A, D, H, X
from mtah.data import TrackingData
from mtah.labels import weak
from mtah.labels.masking import mask_ball, mask_ball_by_state
from mtah.pipeline import engine
from mtah.pipeline.ablation import derive_bands


def _toy(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    players = rng.uniform(0, 60, (n, 4, 2)).astype(np.float32)
    ball = rng.uniform(0, 60, (n, 2)).astype(np.float32)
    ball[5_000:5_400] = np.nan                       # one natural gap
    period = np.ones(n, np.int8)
    labels = np.full(n, H, np.int8)
    for a in range(1_000, n, 4_000):                 # dead segments of 1,500 frames
        labels[a:a + 1_500] = D
    return TrackingData("toy", 25, (105.0, 68.0), np.arange(n), period, players, ball,
                        np.array([0, 0, 1, 1]), labels, {})


def _old_mask_ball(data, rng, target_fraction, lengths):
    """The Adım 2 implementation, kept verbatim as a regression reference."""
    out = data.copy()
    vis = ~np.isnan(data.ball[:, 0])
    candidates = np.flatnonzero(vis & (data.period > 0))
    goal = int(target_fraction * len(candidates))
    masked = np.zeros(data.n_frames, bool)
    while masked.sum() < goal:
        start = int(rng.choice(candidates))
        length = int(rng.choice(lengths))
        stop = min(data.n_frames, start + length)
        masked[start:stop] |= vis[start:stop] & (data.period[start:stop] > 0)
    out.ball[masked] = np.nan
    return out, masked


def test_uniform_masking_is_unchanged():
    data, lengths = _toy(), np.array([50, 120, 400])
    _, new = mask_ball(data, np.random.default_rng(3), 0.4, lengths)
    _, old = _old_mask_ball(data, np.random.default_rng(3), 0.4, lengths)
    assert np.array_equal(new, old)


def test_state_aware_masking_hits_both_rates():
    data, lengths = _toy(), np.array([50, 120, 400])
    out, masked = mask_ball_by_state(data, np.random.default_rng(1), data.labels, 0.58, 0.32, lengths)
    vis = ~np.isnan(data.ball[:, 0])
    dead, live = vis & (data.labels == D), vis & (data.labels != D)
    assert masked[dead].mean() >= 0.58 and masked[dead].mean() < 0.75
    assert masked[live].mean() >= 0.32 and masked[live].mean() < 0.45
    assert not masked[~vis].any()                      # only visible frames are newly masked
    assert np.isnan(out.ball[masked]).all() and not np.isnan(data.ball[masked]).any()


def test_run_cores_erodes_and_filters():
    m = np.zeros(100, bool)
    m[10:40] = True                                    # 30 long -> kept, eroded by 5
    m[60:70] = True                                    # 10 long -> too short
    core = weak.run_cores(m, 20, 5)
    assert core[15:35].all() and not core[10:15].any() and not core[35:40].any()
    assert not core[60:70].any()


def test_tempo_dead_anchor_only_in_slow_core_and_never_over_a_team_anchor():
    n, fps = 2_000, 25
    t = np.arange(n) / fps
    players = np.zeros((n, 22, 2), np.float32)
    players[:, :, 0] = (3.0 * t)[:, None] + np.arange(22)[None, :]       # everyone runs 3 m/s
    players[800:1_400, :, 0] = players[800, :, 0]                         # 24 s standing still
    data = SimpleNamespace(players=players, fps=fps)
    feats = pd.DataFrame({"ball_vis": np.ones(n), "ball_speed": np.full(n, 5.0)})
    period = np.ones(n, np.int8)
    masks = weak.dead_anchor_masks(feats, period, {"d_tempo": True}, data, fps)
    core = masks["d_tempo"]
    assert core[800 + 75 + 15:1_400 - 75 - 15].all()                    # inside, after the 3 s cuts
    assert not core[:800].any() and not core[1_400:].any()


def test_band_floor_widens_a_tight_seed_spread():
    rows = [{"dead_share": 0.320 + 0.001 * i, "n_possessions": 300 + i, "home_share": 0.60} for i in range(5)]
    ac = {"band_stats": ["dead_share", "n_possessions", "home_share"], "band_sigmas": 2.0,
          "band_floor": {"dead_share": 0.03, "n_possessions": 20, "home_share": 0.025}}
    bands = derive_bands(rows, ac)
    assert np.isclose(bands["dead_share"][1] - bands["dead_share"][0], 0.06)
    assert np.isclose(bands["n_possessions"][1] - bands["n_possessions"][0], 40)
    assert np.isclose(bands["home_share"][0], 0.575)


def _source(name, n_rows, share, d_frac, d_share=None):
    y = np.where(np.arange(n_rows) < int(d_frac * n_rows), D, H).astype(np.int8)
    X_ = pd.DataFrame({"f": np.arange(n_rows, dtype=np.float32)})
    src = engine.TrainSource(name, None, X_, y, np.ones(n_rows, bool), np.ones(n_rows, bool),
                             name, np.array([1]), share=share, d_share=d_share)
    return src, (X_, y, np.ones(n_rows, bool))


def test_source_weights_follow_shares_and_d_share():
    cfg = {"seed": 2027, "masking": {"per_seed": True}, "lgbm": {}}
    bar, mbar = _source("bar", 1_000, 0.5, 0.8, d_share=0.35)
    dfl, mdfl = _source("dfl", 4_000, 0.5, 0.45)
    bar._memo[engine.mask_seed(cfg, 0)] = mbar
    dfl._memo[engine.mask_seed(cfg, 0)] = mdfl
    X_, y, _, w, counts = engine._stack([bar, dfl], cfg, 0, ["f"])
    assert counts == {"bar": 1_000, "dfl": 4_000}
    w_bar, w_dfl = w[:1_000], w[1_000:]
    assert np.isclose(w_bar.sum(), w_dfl.sum())                         # equal source shares
    y_bar = y[:1_000]
    assert np.isclose(w_bar[y_bar == D].sum() / w_bar.sum(), 0.35)       # D share held at 0.35
    assert np.allclose(w_dfl, w_dfl[0])                                  # true labels: no rebalancing
    assert engine.mask_seed(cfg, 3) == 2027 + 3000
    assert engine.mask_seed({"seed": 2027, "masking": {}}, 3) == 2027
