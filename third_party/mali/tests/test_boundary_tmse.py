"""Synthetic-only isolation and leakage checks for dense H/A-edge smoothing masks."""

from pathlib import Path
import importlib

import pytest
import torch
import torch.nn.functional as F

from mtah import config
from mtah.constants import H, A, D
from mtah.models import seqnet


def test_disabled_path_is_identical_and_returns_original_tensor():
    d = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3).requires_grad_()
    y = torch.tensor([[H, A, D, H], [A, H, A, D]])
    known = torch.ones_like(y, dtype=torch.bool)
    assert seqnet.mask_team_boundary_tmse(d, y, known, seqnet.DEFAULTS) is d
    assert seqnet.mask_team_boundary_tmse(d, y, known, {}) is d


def test_only_dense_known_team_edges_are_masked_and_dead_padding_unknown_bar_are_unchanged():
    y = torch.tensor([[H, H, A, D, A, H, A], [H, A, H, A, H, A, H]])
    known = torch.ones_like(y, dtype=torch.bool)
    known[0, 5] = False
    d = torch.ones((2, 6, 3))
    result = seqnet.mask_team_boundary_tmse(d, y, known, {"tmse_team_boundary_radius": 0},
                                           torch.tensor([True, False]))
    expected = d.clone()
    expected[0, 1] = 0
    assert torch.equal(result, expected)
    # The normalization remains all 2*6*3 entries, not only unmasked entries.
    assert result.mean().item() == pytest.approx(33 / 36)
    exchanged = torch.where(y < 2, 1 - y, y)
    assert torch.equal(result, seqnet.mask_team_boundary_tmse(d, exchanged, known,
                       {"tmse_team_boundary_radius": 0}, torch.tensor([True, False])))


@pytest.mark.parametrize("weighted", [False, True])
def test_only_tmse_component_changes_and_gradient_matches_removed_edges(weighted):
    torch.manual_seed(3)
    logits = torch.randn((2, 6, 3), requires_grad=True)
    bnd = torch.randn((2, 6), requires_grad=True)
    y = torch.tensor([[H, H, A, A, D, H], [A, H, A, D, H, A]])
    known = torch.ones_like(y, dtype=torch.bool)
    bt = torch.zeros_like(y, dtype=torch.float32)
    kw = dict(w=torch.tensor([[1., 2., 1., 1., 1., 1.], [3., 1., 1., 1., 1., 1.]]),
              bnd_on=torch.tensor([True, False])) if weighted else {}
    p = dict(seqnet.DEFAULTS, lambda_tmse=.3)
    base, parts0 = seqnet.loss_fn(logits, bnd, y, known, bt, p, **kw)
    candidate, parts1 = seqnet.loss_fn(logits, bnd, y, known, bt,
                                    dict(p, tmse_team_boundary_radius=0), **kw)
    for key in ("ce", "iou", "bnd"):
        assert parts0[key] == parts1[key]
    assert candidate < base
    lp = F.log_softmax(logits, dim=-1)
    d = torch.clamp((lp[:, 1:] - lp[:, :-1].detach()) ** 2, max=p["tmse_tau"] ** 2)
    kept = seqnet.mask_team_boundary_tmse(d, y, known, dict(p, tmse_team_boundary_radius=0), kw.get("bnd_on"))
    if weighted:
        w = kw["w"].mean(1)
        expected = .3 * ((d - kept).mean((1, 2)) * w).sum() / w.sum()
    else:
        expected = .3 * (d - kept).mean()
    assert torch.allclose(base - candidate, expected, atol=1e-6)
    grad_actual = torch.autograd.grad(base - candidate, (logits, bnd), retain_graph=True)
    grad_expected = torch.autograd.grad(expected, logits)[0]
    assert torch.allclose(grad_actual[0], grad_expected, atol=1e-7)
    assert torch.equal(grad_actual[1], torch.zeros_like(bnd))


def test_sparse_bar_only_loss_is_bit_identical_with_mask_enabled():
    logits = torch.tensor([[[.8, .1, -.1], [.1, .9, -.2], [.9, .1, -.3]]], requires_grad=True)
    y = torch.tensor([[H, A, H]])
    args = (logits, torch.zeros((1, 3)), y, torch.ones_like(y, dtype=torch.bool), torch.zeros((1, 3)))
    kw = dict(w=torch.tensor([2.]), bnd_on=torch.tensor([False]))
    a, pa = seqnet.loss_fn(*args, seqnet.DEFAULTS, **kw)
    b, pb = seqnet.loss_fn(*args, dict(seqnet.DEFAULTS, tmse_team_boundary_radius=0), **kw)
    assert torch.equal(a, b) and pa == pb


def test_boundary_configs_differ_only_in_mask_and_preserve_augmented_backbone():
    from mtah.validation.splits import cache_tag
    baseline = config.load("ev_bmask_base_rf0")
    old = config.load("ev_ha_swap_rf0")
    candidate = config.load("ev_bmask_team0_rf0")
    assert candidate["seqnet"].pop("tmse_team_boundary_radius") == 0
    assert baseline["seqnet"]["lambda_tmse"] == .3
    for cfg in (baseline, old, candidate):
        assert cache_tag(cfg) == cache_tag(config.load("main_rf0"))
        assert cfg["adapt"]["external_team_copies"] == "swap" and not cfg["adapt"].get("short_team")
        cfg.pop("name")
    assert baseline == old == candidate


def test_boundary_plan_compares_candidate_with_ha_reference(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    runner = importlib.import_module("run_day_screen")
    monkeypatch.setattr(runner, "scores", lambda x: x)
    rows = dict(zip(runner.BOUNDARY_TRIALS, [{"dfl_v1": .7, "dfl_v2b": .68, "metrica": .65},
                                          {"dfl_v1": .71, "dfl_v2b": .67, "metrica": .66}]))
    result = runner.compare("boundary-mask", rows, "synthetic")
    assert result["delta"] == pytest.approx({"dfl_v1": .01, "dfl_v2b": -.01, "metrica": .01})
