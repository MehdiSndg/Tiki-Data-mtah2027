"""Synthetic invariance, capacity, missingness and full-token training checks."""

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from mtah import config
from mtah.models import seqnet
from mtah.models.player_set import TeamSetPool, compact, SET_SHAPE


def test_pool_is_permutation_invariant_and_all_missing_safe():
    torch.manual_seed(1)
    pool = TeamSetPool(9, 8)
    tok = torch.randn(2, 4, 24, 9)
    tok[..., 4] = torch.tensor([1.] * 12 + [-1.] * 12)
    tok[..., -1] = 1
    tok[:, :, 2, -1] = 0
    tok[:, :, 2, :-1] = torch.nan
    out = pool(tok)
    perm = torch.randperm(24)
    assert torch.allclose(out, pool(tok[:, :, perm]), atol=1e-6)
    assert torch.isfinite(out).all()
    assert torch.equal(pool(torch.zeros_like(tok)), torch.zeros_like(out))
    out.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in pool.parameters())


def test_compaction_keeps_both_teams_keepers_and_nearest_players_without_slot_bias():
    tok = np.zeros((2, 30, 9), np.float32)
    tok[..., 4] = np.r_[np.ones(15), -np.ones(15)]
    tok[..., -1] = 1
    tok[..., 6] = np.arange(30)
    tok[:, [14,29], 5] = 1
    out = compact(tok)
    assert out.shape == (2, 24, 9)
    assert np.all(out[:, :12, 4] == 1) and np.all(out[:, 12:, 4] == -1)
    assert np.all(out[..., 5].sum(1) == 2)
    assert set(out[0, :12, 6]) == set(range(11)) | {14}
    perm = np.random.default_rng(4).permutation(30)
    np.testing.assert_array_equal(out, compact(tok[:, perm]))
    tok[:, :15, -1] = 0
    missing = compact(tok)
    assert np.all(missing[:, :12] == 0)


def test_small_set_gru_training_is_deterministic_and_normalized():
    rng = np.random.default_rng(3)
    n = 80
    tok = rng.normal(size=(n, 24, 9)).astype(np.float32)
    tok[..., 4] = np.r_[np.ones(12), -np.ones(12)]
    tok[..., -1] = 1
    x = np.c_[rng.normal(size=(n, 4)), tok.reshape(n, -1)].astype(np.float32)
    y = np.resize([0,0,1,1,2], n).astype(np.int64)
    known = np.ones(n, bool)
    p = dict(seqnet.DEFAULTS, use_tokens=True, token_pool="deepsets_team", token_shape=SET_SHAPE,
             token_dim=4, hidden=16, gru_hidden=8, dilations=[], dropout=0., chunk=32,
             chunk_step=16, batch=4, epochs=2)
    seqs = [(x,y,known,1.,True), (x,y,known,.5,False)]
    a = seqnet.predict(seqnet.train(seqs,p,9),x)
    b = seqnet.predict(seqnet.train(seqs,p,9),x)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(a.sum(1),1.,atol=1e-6)


def test_set_configs_preserve_hypothesis_controls_and_frozen_booster():
    from mtah.validation.splits import cache_tag
    base, gru, pooled = [config.load(n) for n in ("ev_set_base_rf0","ev_set_gru_rf0","ev_set_pool_rf0")]
    for cfg in (base,gru,pooled):
        assert cfg["seqnet"]["n_seeds"] == 2 and cfg["adapt"]["external_team_copies"] == "swap"
        assert cfg["seqnet"].get("tmse_weak_support","all") == "all"
        assert cfg["seqnet"].get("tmse_team_boundary_radius",-1) == -1
        assert cache_tag(cfg) == cache_tag(config.load("main_rf0"))
        cfg.pop("name")
    assert gru["seqnet"]["dilations"] == pooled["seqnet"]["dilations"] == []
    assert pooled["seqnet"].pop("use_tokens") is True
    assert pooled["seqnet"].pop("token_pool") == "deepsets_team"
    assert pooled["seqnet"].pop("token_dim") == 16
    assert pooled == gru
    gru["seqnet"].pop("dilations")
    assert base == gru


def test_fixed_posterior_mixture_is_normalized_equivariant_and_does_not_mutate(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"tools"))
    mix = importlib.import_module("run_member_blend").mix_probabilities
    a = np.array([[.7,.2,.1],[.1,.2,.7]])
    b = np.array([[.2,.7,.1],[.3,.2,.5]])
    before = a.copy()
    out = mix(a,b)
    np.testing.assert_allclose(out, (a+b)*.5)
    np.testing.assert_allclose(mix(a[:,[1,0,2]],b[:,[1,0,2]]),out[:,[1,0,2]])
    np.testing.assert_array_equal(a,before)
    with pytest.raises(ValueError):
        mix(a,b[:1])
