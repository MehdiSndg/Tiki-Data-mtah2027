import numpy as np

from mtah.balltrack import BROADCAST, DETECTOR, INTERP, NONE, TACTICAL, fuse_ball

PITCH = (105.0, 68.0)
nan = np.nan


def arrays(n):
    return np.full((n, 2), nan), np.full((n, 2), nan), np.full((n, 2), nan), np.zeros(n), np.zeros(n, bool)


def test_tactical_wins_even_against_the_detector():
    tac, bro, det, conf, spot = arrays(3)
    tac[:] = [0.5, 0.5]
    bro[:] = [0.6, 0.5]
    det[:] = [0.5, 0.4]                       # an aerial ball: the image puts it elsewhere
    conf[:] = 0.9
    out = fuse_ball(tac, bro, det, conf, spot, PITCH)
    assert np.allclose(out["X"], 0.5) and np.all(out["source"] == TACTICAL)


def test_broadcast_then_detector_fill_a_tactical_gap():
    tac, bro, det, conf, spot = arrays(4)
    tac[0] = [0.50, 0.5]
    bro[1] = [0.505, 0.5]
    det[2], conf[2] = [0.51, 0.5], 0.6
    out = fuse_ball(tac, bro, det, conf, spot, PITCH)
    assert out["source"][:3].tolist() == [TACTICAL, BROADCAST, DETECTOR]
    assert out["source"][3] == NONE


def test_detector_jump_rejected():
    tac, bro, det, conf, spot = arrays(6)
    tac[0] = [0.1, 0.5]
    det[5], conf[5] = [0.6, 0.5], 0.9         # 52 m in 0.2 s
    out = fuse_ball(tac, bro, det, conf, spot, PITCH)
    assert out["source"][5] == NONE


def test_spot_box_needs_continuity():
    tac, bro, det, conf, spot = arrays(30)
    tac[0] = [11 / 105, 0.5]                  # the ball was seen on the penalty spot
    det[1:30], conf[1:30], spot[1:30] = [11 / 105, 0.5], 0.5, True
    out = fuse_ball(tac, bro, det, conf, spot, PITCH)
    assert np.all(out["source"][1:] == DETECTOR)
    tac2 = tac.copy()
    tac2[0] = [50 / 105, 0.5]                 # the ball was at midfield 1.2 s before: the spot box is the paint
    out = fuse_ball(tac2, bro, det, conf, spot, PITCH)
    assert np.all(out["source"][1:] != DETECTOR)


def test_short_gap_interpolated_long_gap_not():
    tac, bro, det, conf, spot = arrays(100)
    tac[0], tac[20], tac[99] = [0.2, 0.5], [0.3, 0.5], [0.4, 0.5]
    out = fuse_ball(tac, bro, det, conf, spot, PITCH)
    assert np.all(out["source"][1:20] == INTERP) and np.isclose(out["X"][10], 0.25)
    assert np.all(out["source"][21:99] == NONE)


def test_fuse_deterministic():
    rng = np.random.RandomState(0)
    tac, bro, det, conf, spot = arrays(500)
    keep = rng.rand(500) < 0.6
    tac[keep] = rng.rand(keep.sum(), 2)
    det[:] = rng.rand(500, 2)
    conf[:] = rng.rand(500)
    a, b = fuse_ball(tac, bro, det, conf, spot, PITCH), fuse_ball(tac, bro, det, conf, spot, PITCH)
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)


def test_online_tracking_ignores_a_far_confident_distractor():
    from mtah.balltrack import track_detections
    n = 40
    frame = np.r_[np.arange(n), np.arange(n)]
    x = np.r_[10 + 0.4 * np.arange(n), np.full(n, 80.0)]          # ball rolls at 10 m/s; a 'ball' 70 m away
    y = np.r_[np.full(n, 30.0), np.full(n, 30.0)]
    conf = np.r_[np.full(n, 0.4), np.full(n, 0.9)]
    conf[:3] = 0.95                                                 # the track starts on the real ball
    X, Y, C = track_detections(frame, x, y, conf, n)
    assert np.allclose(X, 10 + 0.4 * np.arange(n))
    assert np.all(C > 0)


def test_online_tracking_restarts_after_losing_the_ball():
    from mtah.balltrack import track_detections, LOST_FRAMES
    n = 20 + LOST_FRAMES + 20
    frame = np.r_[np.arange(20), np.arange(20 + LOST_FRAMES, n)]
    x = np.r_[np.full(20, 10.0), np.full(20, 90.0)]
    X, _, _ = track_detections(frame, x, np.full(40, 30.0), np.full(40, 0.8), n)
    assert np.isnan(X[20:20 + LOST_FRAMES]).all() and np.allclose(X[-20:], 90.0)
