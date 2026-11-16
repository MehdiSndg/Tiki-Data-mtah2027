"""Synthetic checks for a weak-label support ablation, not an unknown-label bug fix."""

import importlib
from pathlib import Path

import pytest
import torch

from mtah import config
from mtah.models import seqnet


def test_default_and_dense_pretraining_are_noops():
    d = torch.ones((2, 4, 3))
    known = torch.tensor([[True, True, False, False, True]] * 2)
    assert seqnet.weak_support_tmse(d, known, {}) is d
    assert seqnet.weak_support_tmse(d, known, seqnet.DEFAULTS, torch.tensor([True, False])) is d
    for mode in ("scale", "known"):
        assert seqnet.weak_support_tmse(d, known, {"tmse_weak_support": mode}) is d


def test_known_support_and_scale_preserve_dense_windows_and_equal_total_weak_gate_weight():
    d = torch.ones((2, 4, 3))
    known = torch.tensor([[True, True, False, False, True]] * 2)
    dense = torch.tensor([True, False])
    scale = seqnet.weak_support_tmse(d, known, {"tmse_weak_support": "scale"}, dense)
    masked = seqnet.weak_support_tmse(d, known, {"tmse_weak_support": "known"}, dense)
    assert torch.equal(scale[0], d[0]) and torch.equal(masked[0], d[0])
    assert torch.equal(masked[1, 0], d[1, 0]) and torch.equal(masked[1, 1:], torch.zeros((3, 3)))
    assert torch.equal(scale[1], torch.full((4, 3), .25))
    assert torch.equal(scale.sum(1), masked.sum(1))


@pytest.mark.parametrize("mode", ["known", "scale"])
def test_only_tmse_changes_for_weighted_mixture_and_dense_only_loss_is_bit_identical(mode):
    torch.manual_seed(5)
    logits = torch.randn((2, 5, 3), requires_grad=True)
    bnd = torch.randn((2, 5), requires_grad=True)
    y = torch.tensor([[0, 1, 1, 2, 0], [0, 1, 1, 2, 0]])
    known = torch.tensor([[True] * 5, [True, True, False, False, True]])
    args = (logits, bnd, y, known, torch.zeros((2, 5)))
    p = dict(seqnet.DEFAULTS)
    candidate_p = dict(p, tmse_weak_support=mode)
    kw = dict(w=torch.tensor([[1.] * 5, [2., 3., 0., 0., 2.]]), bnd_on=torch.tensor([True, False]))
    base, parts0 = seqnet.loss_fn(*args, p, **kw)
    candidate, parts1 = seqnet.loss_fn(*args, candidate_p, **kw)
    assert candidate < base and parts1["tmse"] < parts0["tmse"]
    for field in ("ce", "iou", "bnd"):
        assert parts0[field] == parts1[field]
    g, gb = torch.autograd.grad(base - candidate, (logits, bnd))
    assert torch.equal(g[0], torch.zeros_like(g[0]))
    assert torch.equal(gb, torch.zeros_like(gb))
    a, pa = seqnet.loss_fn(*args, p)
    b, pb = seqnet.loss_fn(*args, candidate_p)
    assert torch.equal(a, b) and pa == pb


def test_all_known_support_is_identical_and_zero_support_has_no_nans():
    d = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    for mode in ("scale", "known"):
        assert torch.equal(seqnet.weak_support_tmse(d, torch.ones((2, 5), dtype=torch.bool),
                           {"tmse_weak_support": mode}, torch.tensor([False, False])), d)
        result = seqnet.weak_support_tmse(d, torch.zeros((2, 5), dtype=torch.bool),
                                          {"tmse_weak_support": mode}, torch.tensor([False, False]))
        assert torch.equal(result, torch.zeros_like(d)) and torch.isfinite(result).all()


def test_configs_keep_boundary_mask_off_and_isolate_support_on_same_ha_backbone():
    from mtah.validation.splits import cache_tag
    rows = [config.load(n) for n in ("ev_support_base_rf0", "ev_support_scale_rf0", "ev_support_known_rf0")]
    old = config.load("ev_ha_swap_rf0")
    for cfg in rows:
        assert cfg["seqnet"].get("tmse_team_boundary_radius", -1) == -1
        assert cfg["adapt"]["external_team_copies"] == "swap" and not cfg["adapt"].get("short_team")
        assert cache_tag(cfg) == cache_tag(config.load("main_rf0"))
        cfg.pop("name")
        cfg["seqnet"].pop("tmse_weak_support", None)
    old.pop("name")
    assert rows[0] == rows[1] == rows[2] == old


def test_comparison_separates_support_from_strength(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    runner = importlib.import_module("run_day_screen")
    monkeypatch.setattr(runner, "scores", lambda x: x)
    rows = dict(zip(runner.SUPPORT_TRIALS, [{"dfl_v1": .70, "dfl_v2b": .68, "metrica": .65},
                                         {"dfl_v1": .72, "dfl_v2b": .70, "metrica": .67},
                                         {"dfl_v1": .73, "dfl_v2b": .71, "metrica": .68}]))
    result = runner.compare("weak-support", rows, "synthetic")
    assert result["known_minus_reference"]["metrica"] == pytest.approx(.03)
    assert result["known_minus_scale"]["metrica"] == pytest.approx(.01)
