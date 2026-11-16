import numpy as np

from mtah import grumodel


def test_chunked_prediction_covers_every_graded_frame_and_leaves_the_rest_at_half():
    import torch
    torch.manual_seed(0)
    net = grumodel._net(3).eval()
    n = 4000
    F = np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32)
    g = np.zeros(n, bool)
    g[100:3900] = True
    p = grumodel._predict_one(net, F, g, "cpu")
    assert np.all(p[~g] == 0.5) and np.all((p[g] > 0) & (p[g] < 1)) and not np.any(np.isnan(p))


def test_training_learns_an_obvious_signal():
    rng = np.random.default_rng(0)
    n = 3000
    y = (np.arange(n) // 300 % 2).astype(np.float32)
    F = np.c_[y * 2 - 1 + rng.normal(0, 0.3, n), rng.normal(size=n)].astype(np.float32)
    old = grumodel.EPOCHS
    grumodel.EPOCHS = 3
    try:
        net = grumodel.train([(F, y, np.ones(n, np.float32), np.ones(n, bool))], 0, "cpu")
    finally:
        grumodel.EPOCHS = old
    p = grumodel._predict_one(net, F, np.ones(n, bool), "cpu")
    assert np.mean((p > 0.5) == (y > 0.5)) > 0.95


def test_two_head_training_learns_both_targets_and_predict_returns_the_team_head():
    rng = np.random.default_rng(0)
    n = 3000
    y = (np.arange(n) // 300 % 2).astype(np.float32)
    b = (np.arange(n) // 150 % 2).astype(np.float32)
    F = np.c_[y * 2 - 1 + rng.normal(0, 0.3, n), b * 2 - 1 + rng.normal(0, 0.3, n)].astype(np.float32)
    old = grumodel.EPOCHS
    grumodel.EPOCHS = 3
    try:
        net = grumodel.train([(F, np.c_[y, b], np.ones((n, 2), np.float32), np.ones(n, bool))], 0, "cpu",
                             make_net=lambda nf: grumodel.widen(grumodel._net, nf))
    finally:
        grumodel.EPOCHS = old
    p = grumodel._predict_one(net, F, np.ones(n, bool), "cpu")
    assert p.shape == (n, 2)
    assert np.mean((p[:, 0] > 0.5) == (y > 0.5)) > 0.95 and np.mean((p[:, 1] > 0.5) == (b > 0.5)) > 0.95
    model = grumodel.GRUEnsemble([net], np.zeros(2), np.ones(2), "cpu", heads=2)
    m = {"half": np.ones(n, np.int8)}
    old_mf = grumodel.model_features
    grumodel.model_features = lambda arch, m: F
    try:
        q = grumodel.predict(model, m)
    finally:
        grumodel.model_features = old_mf
    assert q.shape == (n,) and np.allclose(q, p[:, 0], atol=1e-6)


def test_jittered_windows_cover_the_tail_of_each_run():
    n = grumodel.WIN * 3 + 37
    F, y, w, g = np.zeros((n, 1), np.float32), np.zeros(n, np.float32), np.ones(n, np.float32), np.ones(n, bool)
    plain = grumodel._windows(F, y, w, g)
    tail = grumodel._windows(F, y, w, g, offset=0, tail=True)
    assert len(tail) == len(plain) + 1 and len(tail[-1][0]) == grumodel.WIN        # one more window, ending at the run's end
    off = grumodel._windows(F, y, w, g, offset=100, tail=True)
    assert len(off[0][0]) == grumodel.WIN and all(len(x[0]) <= grumodel.WIN for x in off)


def test_big_and_wide_members_match_the_tested_nets():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
    from functools import partial
    from mtah import grumodel as G
    from p9am_big_gru import big_net
    n = lambda net: sum(p.numel() for p in net.parameters())
    assert n(G.NETS["big"](10)) == n(big_net(10))
    assert n(G.NETS["gru"](10)) == n(G._net(10, hidden=128, layers=2))
    assert n(G.NETS["tcn128"](10)) == n(partial(G._tcn, dilations=(1, 2, 4, 8, 16, 32, 64, 128))(10))
