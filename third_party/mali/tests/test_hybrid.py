"""Gün 2: hybrid pipeline helpers and the temporal net (skipped without torch)."""

import numpy as np
import pandas as pd
import pytest

from mtah.constants import A, D, H, X
from mtah.pipeline.hybrid import log_pool, robust_scale, step_view, to_frames


def test_robust_scale_handles_nan_and_constant_columns():
    col = np.r_[np.arange(1.0, 9.0), 100.0]                       # median 5, IQR 4
    X = np.stack([col, np.full(9, 5.0), np.full(9, np.nan)], axis=1)
    Z = robust_scale(X)
    assert np.isfinite(Z).all() and Z.dtype == np.float32
    assert np.all(Z[:, 1] == 0) and np.all(Z[:, 2] == 0)
    assert Z[-1, 0] == 5.0 and np.isclose(Z[0, 0], -1.0)                  # clipped / scaled


def test_step_view_block_means_and_centre_labels():
    n = 100
    feats = pd.DataFrame({"f": np.arange(n, dtype=float)})
    probs = np.full((n, 3), 1 / 3)
    labels = np.array([H] * 50 + [D] * 25 + [X] * 25, np.int8)
    (x, y, known, (a, b)), = step_view(feats, ["f"], probs, labels, [(0, n)], 5)
    assert x.shape == (20, 4) and (a, b) == (0, n)
    assert y[:10].tolist() == [H] * 10 and y[10:15].tolist() == [D] * 5 and not known[15:].any()


def test_to_frames_interpolates_and_normalises():
    sp = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    f = to_frames(sp, 0, 10, 5)
    assert f.shape == (10, 3) and np.allclose(f.sum(1), 1)
    assert f[0, 0] == 1.0 and f[-1, 1] == 1.0 and 0 < f[5, 0] < 1


def test_log_pool_limits():
    a = np.array([[0.7, 0.2, 0.1]])
    b = np.array([[0.1, 0.1, 0.8]])
    assert np.allclose(log_pool(a, b, 1.0), a, atol=1e-6)
    assert np.allclose(log_pool(a, b, 0.0), b, atol=1e-6)
    m = log_pool(a, b, 0.5)
    assert np.isclose(m.sum(), 1, atol=1e-5) and m[0, 2] > a[0, 2] and m[0, 2] < b[0, 2]


def test_seqnet_trains_deterministically_and_learns_a_toy_task():
    torch = pytest.importorskip("torch")
    from mtah.models import seqnet
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(3):
        y = np.repeat(rng.integers(0, 3, 40), 25)                        # long constant spells
        x = (np.eye(3)[y] + rng.normal(0, 0.8, (len(y), 3))).astype(np.float32)
        seqs.append((x, y.astype(np.int64), np.ones(len(y), bool)))
    p = dict(seqnet.DEFAULTS, epochs=4, chunk=128, chunk_step=64, hidden=32, gru_hidden=16,
             dilations=(1, 2, 4))
    m1 = seqnet.train(seqs, p, 7)
    m2 = seqnet.train(seqs, p, 7)
    p1, p2 = seqnet.predict(m1, seqs[0][0]), seqnet.predict(m2, seqs[0][0])
    assert np.array_equal(p1, p2)                                         # bit-identical rerun
    assert np.mean(p1.argmax(1) == seqs[0][1]) > 0.8


def test_adapt_mixed_weights_fix_bar_share_and_dead_share():
    from mtah.pipeline.adapt import mixed
    rng = np.random.default_rng(1)
    dfl = [(np.zeros((100, 2), np.float32), rng.integers(0, 3, 100), np.ones(100, bool), (0, 500))]
    y = np.array([D] * 82 + [H] * 12 + [A] * 6)
    bar = [(np.zeros((100, 2), np.float32), y, np.ones(100, bool), (0, 500))]
    seqs, w_bar = mixed(dfl, bar, 0.3, d_share=0.35)
    wb = seqs[1][3]
    assert wb.shape == (100,) and seqs[1][4] is False and seqs[0][3] == 1.0
    assert np.isclose(wb[y == D].sum() / wb.sum(), 0.35)
    assert np.isclose(wb.sum() / (wb.sum() + 100.0), 0.3)                  # BAR share unchanged
    seqs0, _ = mixed(dfl, bar, 0.3)
    assert np.ndim(seqs0[1][3]) == 0                                       # old scalar path


def test_seqnet_accepts_per_step_weights():
    pytest.importorskip("torch")
    from mtah.models import seqnet
    rng = np.random.default_rng(0)
    y = np.repeat(rng.integers(0, 3, 20), 25).astype(np.int64)
    x = (np.eye(3)[y] + rng.normal(0, 0.5, (len(y), 3))).astype(np.float32)
    k = np.ones(len(y), bool)
    seqs = [(x, y, k, 1.0, True), (x, y, k, np.where(y == 2, 0.5, 2.0).astype(np.float32), False)]
    p = dict(seqnet.DEFAULTS, epochs=2, chunk=128, chunk_step=64, hidden=16, gru_hidden=8, dilations=(1, 2))
    m = seqnet.train(seqs, p, 3)
    assert seqnet.predict(m, x).shape == (len(y), 3)


def test_bar_team_weight_lambda_flags_and_training():
    from mtah.pipeline.adapt import mixed
    rng = np.random.default_rng(3)
    y = np.repeat(rng.integers(0, 3, 20), 25).astype(np.int64)
    x = (np.eye(3)[y] + rng.normal(0, 0.5, (len(y), 3))).astype(np.float32)
    k = np.ones(len(y), bool)
    dfl = [(x, y, k, (0, len(y)))]
    bar = [(x, y, k, (0, len(y)))]
    s1, _ = mixed(dfl, bar, 0.3, 0.55, True)
    assert len(s1[0]) == 5 and len(s1[1]) == 5                       # lambda 1: old tuples
    s5, _ = mixed(dfl, bar, 0.3, 0.55, 0.5)
    assert s5[0][5] == 1.0 and s5[1][5] == 0.5
    s0, _ = mixed(dfl, bar, 0.3, None, False)
    assert s0[1][5] == 0.0
    pytest.importorskip("torch")
    from mtah.models import seqnet
    p = dict(seqnet.DEFAULTS, epochs=2, chunk=128, chunk_step=64, hidden=16, gru_hidden=8, dilations=(1, 2))
    m = seqnet.train(s5, p, 5)
    assert np.isfinite(seqnet.predict(m, x)).all()


def test_player_tokens_shape_and_mask():
    from mtah import config
    from mtah.data import TrackingData
    from mtah.pipeline.hybrid import TOKEN_SHAPE, player_tokens
    n = 500
    rng = np.random.default_rng(0)
    players = rng.uniform(0, 60, (n, 6, 2)).astype(np.float32)
    players[:, 0, 0] = 5.0                                         # home keeper deep at low x
    players[:, 3, 0] = 100.0                                       # away keeper deep at high x
    players[100:200, 2] = np.nan                                   # a missing slot
    ball = rng.uniform(10, 50, (n, 2)).astype(np.float32)
    ball[300:] = np.nan
    data = TrackingData("toy", 25, (105.0, 68.0), np.arange(n), np.ones(n, np.int8), players, ball,
                        np.array([0, 0, 0, 1, 1, 1]), None, {})
    toks = player_tokens(data, config.load("main"), [(0, n)], 5)
    t = toks[0].reshape(100, *TOKEN_SHAPE)
    assert t.shape == (100, 48, 9)
    assert np.all(t[:, 6:, :] == 0)                                # padding
    assert np.all(t[20:40, 2, 8] == 0) and np.all(t[:20, 2, 8] == 1)   # missing slot masked
    assert np.all(t[60:, :6, 6:8] == 0)                            # no ball -> no relative offsets
    assert np.all(t[:, 0, 5] == 1) and np.all(t[:, 3, 5] == 1)     # keepers flagged


def test_token_pool_is_permutation_invariant():
    torch = pytest.importorskip("torch")
    from mtah.models import seqnet
    p = dict(seqnet.DEFAULTS, hidden=16, gru_hidden=8, dilations=(1, 2), use_tokens=True,
             token_shape=(6, 9), token_dim=8, token_queries=2)
    seqnet.set_determinism(0)
    model = seqnet.SeqNet(4 + 6 * 9, p).eval()
    rng = np.random.default_rng(1)
    step = rng.normal(size=(1, 30, 4)).astype(np.float32)
    tok = rng.normal(size=(1, 30, 6, 9)).astype(np.float32)
    tok[..., 8] = 1.0
    tok[:, :, 5, 8] = 0.0                                          # one absent slot
    perm = tok[:, :, [3, 0, 5, 1, 4, 2], :]
    x1 = torch.tensor(np.concatenate([step, tok.reshape(1, 30, -1)], -1))
    x2 = torch.tensor(np.concatenate([step, perm.reshape(1, 30, -1)], -1))
    with torch.no_grad():
        a, _ = model(x1)
        b, _ = model(x2)
    assert torch.allclose(a, b, atol=1e-5)


def test_split_heads_compose_a_distribution_and_ignore_the_touch_head():
    """Gün 3 / Adım 2 skeleton: [H, A, D] = alive x team_ctrl; team_touch never changes it."""
    import torch
    from mtah.models import seqnet
    p = dict(seqnet.DEFAULTS, heads="split", hidden=16, gru_hidden=8, dilations=(1, 2))
    torch.manual_seed(0)
    net = seqnet.SeqNet(6, p).to(seqnet.device()).eval()
    x = np.random.default_rng(0).normal(size=(40, 6)).astype(np.float32)
    pr = seqnet.predict(net, x)
    assert pr.shape == (40, 3) and np.allclose(pr.sum(1), 1.0, atol=1e-6)
    with torch.no_grad():
        net.touch.weight.mul_(-3.0)
        net.touch.bias.add_(5.0)
    assert np.allclose(seqnet.predict(net, x), pr)
    tl = net.touch_logits(torch.tensor(x[None], device=seqnet.device()))
    assert tuple(tl.shape) == (1, 40, 3)


def test_split_heads_train_with_touch_targets_deterministically():
    """Gün 3 / Adım 2: team_touch gets its own targets; joint mode ignores `touch` entirely."""
    from mtah.models import seqnet
    rng = np.random.default_rng(3)
    T = 300
    x = rng.normal(size=(T, 5)).astype(np.float32)
    y = (np.arange(T) // 40 % 3).astype(np.int64)
    k = np.ones(T, bool)
    touch = [np.where(rng.uniform(size=T) < 0.2, 1 - (y % 2), y).astype(np.int64)]
    p = dict(seqnet.DEFAULTS, heads="split", hidden=16, gru_hidden=8, dilations=(1, 2), epochs=3,
             chunk=128, chunk_step=64, batch=4)
    m1 = seqnet.train([(x, y, k)], p, 7, touch=touch)
    m2 = seqnet.train([(x, y, k)], p, 7, touch=touch)
    assert np.array_equal(seqnet.predict(m1, x), seqnet.predict(m2, x))
    log = []
    seqnet.train([(x, y, k)], p, 7, touch=touch, log=log)
    assert "touch" in log[-1]
    pj = dict(p, heads="joint")
    a = seqnet.predict(seqnet.train([(x, y, k)], pj, 7), x)
    b = seqnet.predict(seqnet.train([(x, y, k)], pj, 7, touch=touch), x)
    assert np.array_equal(a, b)


def test_adapt_params_let_command_line_override_the_config():
    """Gün 3 bug: --dfl-target ctrl with configs/main.yaml (dfl_target gt35) raised a TypeError."""
    from mtah import config
    from mtah.pipeline import adapt
    cfg = config.load("main")
    ac, p = adapt.params(cfg, {"dfl_target": "ctrl", "d_share": 0.4, "seqnet": {"n_seeds": 2, "heads": "split"}})
    assert ac["dfl_target"] == "ctrl" and ac["d_share"] == 0.4
    assert p["n_seeds"] == 2 and p["heads"] == "split" and p["lambda_tmse"] == 0.3
    ac0, p0 = adapt.params(cfg)
    assert ac0["dfl_target"] == "gt20" and p0["n_seeds"] == 5


def test_fine_branch_learns_sub_step_flips_deterministically():
    """Gün 4 / Sıra 3: the 25 fps branch sees frame-level evidence the 5 Hz step mean blurs."""
    from mtah.models import seqnet
    rng = np.random.default_rng(5)
    n_steps, ff = 240, 4
    yf = np.zeros((n_steps, 5), np.int64)
    for s0 in range(10, n_steps - 10, 23):
        yf[s0, 1:3] = 1                                   # 2-frame A flips inside H steps
    xf = rng.normal(0, 0.1, size=(n_steps, 5, ff)).astype(np.float32)
    xf[..., 0] += (yf == 1) * 3.0                         # frame-level evidence
    x = xf.mean(1)                                        # 5 Hz input: the flip is diluted
    y = np.zeros(n_steps, np.int64)
    k = np.ones(n_steps, bool)
    p = dict(seqnet.DEFAULTS, hidden=16, gru_hidden=8, dilations=(1, 2), epochs=30, chunk=120, chunk_step=60,
             batch=4, fine_dim=ff, fine_hidden=8)
    fine = [(xf, yf)]
    m1 = seqnet.train([(x, y, k)], p, 3, fine=fine)
    m2 = seqnet.train([(x, y, k)], p, 3, fine=fine)
    p1, p2 = seqnet.predict_fine(m1, x, xf), seqnet.predict_fine(m2, x, xf)
    assert p1.shape == (5 * n_steps, 3) and np.allclose(p1.sum(1), 1, atol=1e-5)
    assert np.array_equal(p1, p2)
    flips = yf.reshape(-1) == 1
    assert p1[flips, 1].mean() > 0.5 > p1[~flips, 1].mean()


def test_fine_gate_and_view():
    import pandas as pd
    from mtah.models import seqnet
    from mtah.pipeline import hybrid
    g = seqnet.fine_gate(np.array([0, 0, 0, 1, 1, -1, -1, 1]), 1)
    assert g.tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    n, stride = 50, 5
    feats = pd.DataFrame({c: np.arange(n, dtype=np.float32) for c in hybrid.FINE_COLS})
    lab = np.full(n, 3, np.int8)
    lab[10:20] = 1
    steps = [(np.zeros((8, 2)), None, None, (5, 45))]
    (xf, yf), = hybrid.fine_view(feats, np.full((n, 3), 1 / 3), lab, steps, stride)
    assert xf.shape == (8, 5, hybrid.FINE_DIM) and yf.shape == (8, 5)
    assert (yf.reshape(-1)[5:15] == 1).all() and (yf.reshape(-1)[:5] == -1).all()
