import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.ballimpute import FPS, fill, gap_features, offset_target  # noqa: E402


def toy(n=1000):
    bx, by = np.full(n, 0.5, np.float32), np.full(n, 0.5, np.float32)
    bx[400:] = 0.6
    bx[100:400] = np.nan                     # 12 s gap
    bx[600:650] = np.nan                     # 2 s gap
    by[np.isnan(bx)] = np.nan
    return {"bx": bx, "by": by, "pitch": (100.0, 50.0), "half": np.ones(n, np.int8)}


def test_gap_features_time_since_until_and_jump():
    G = gap_features(toy())
    assert G.shape == (1000, 4)
    assert np.isclose(G[250, 0], 151 / FPS) and np.isclose(G[250, 1], 150 / FPS)   # since last / until next sighting, s
    assert np.isclose(G[250, 2], 10.0) and np.isclose(G[250, 3], 0.0)              # next - last sighting, m
    assert np.allclose(G[50], 0)


def test_offset_target_is_true_ball_minus_last_sighting_on_lost_frames():
    m = toy()
    raw_bx, raw_by = np.full(1000, 0.55, np.float32), np.full(1000, 0.5, np.float32)
    y, w = offset_target(m, raw_bx, raw_by)
    assert np.allclose(y[250], [0.5, 0.0])           # (0.55 - 0.5) * 100 m / 10
    assert w[250] == 1 and w[50] == 0


def test_fill_only_gaps_at_least_min_gap():
    m = toy()
    off = np.zeros((1000, 2), np.float32)
    off[:, 0] = 0.3                                  # +3 m
    f = fill(m, off, min_gap_s=10)
    assert np.isclose(f["bx"][250], 0.53) and np.isnan(f["bx"][620])
    assert np.isnan(m["bx"][250])                    # input untouched
