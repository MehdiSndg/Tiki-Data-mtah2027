"""Synthetic tests: full-resolution duration masks and isolation of the conditional team loss."""

import numpy as np
import pytest

from mtah.constants import H, A, D, X
from mtah.labels.short_team import frame_weights, step_weights


def test_strict_duration_threshold_padding_and_dead_unknown_exclusion():
    y = np.array([H] * 125 + [A] * 25 + [H] * 125 + [D] * 20 + [A] * 125 + [X] * 10)
    w = frame_weights(y, [(0, len(y))], fps=25)
    expected = np.ones(len(y), np.float32)
    expected[113:162] = 2
    assert np.array_equal(w, expected)
    assert (w[np.isin(y, (D, X))] == 1).all()


def test_padding_does_not_cross_played_runs_or_stack_overlapping_weights():
    y = np.array([H] * 125 + [A] * 10 + [D] * 5 + [H] * 125 + [A] * 10 + [H] * 10 + [A] * 125)
    w = frame_weights(y, [(0, 135), (140, len(y))], fps=25)
    assert (w[135:152] == 1).all()
    assert w.max() == 2 and w[130] == 2 and w[270] == 2


def test_step_sampling_uses_full_target_duration_before_chunking():
    y = np.array([H] * 125 + [A] * 124 + [D] * 6)
    x = np.zeros((51, 2), np.float32)
    steps = [(x, None, None, (0, 255))]
    w, = step_weights(steps, y, [(0, 255)], 25, 5, pad_frames=12)
    assert np.array_equal(w, frame_weights(y, [(0, 255)], 25)[2::5])
    assert (w[:22] == 1).all() and (w[23:49] == 2).all() and w[-1] == 1


@pytest.mark.parametrize("weighted", [False, True])
def test_only_team_ce_changes_and_extra_dead_logit_gradient_is_zero(weighted):
    torch = pytest.importorskip("torch")
    from mtah.models import seqnet
    logits = torch.tensor([[[0.2, -0.4, 0.8], [0.6, 0.1, -0.2], [-0.2, 0.8, 0.3], [0.1, 0.3, 0.4]]],
                          requires_grad=True)
    bnd = torch.zeros((1, 4), requires_grad=True)
    y = torch.tensor([[H, A, D, H]])
    known = torch.tensor([[True, True, True, False]])
    bt = torch.tensor([[0., 1., 1., 0.]])
    args = dict(w=torch.tensor([[1., 0.5, 2., 3.]]), bnd_on=torch.tensor([True])) if weighted else {}
    base, p0 = seqnet.loss_fn(logits, bnd, y, known, bt, seqnet.DEFAULTS, **args)
    candidate, p1 = seqnet.loss_fn(logits, bnd, y, known, bt, seqnet.DEFAULTS,
                                  team_weight=torch.full((1, 4), 2.), **args)
    assert candidate > base
    for key in ("tmse", "iou", "bnd"):
        assert p0[key] == p1[key]
    gl, gb = torch.autograd.grad(candidate - base, (logits, bnd))
    assert torch.allclose(gl[..., D], torch.zeros_like(gl[..., D]), atol=1e-7)
    assert torch.equal(gl[:, 2:], torch.zeros_like(gl[:, 2:]))
    assert torch.equal(gb, torch.zeros_like(gb))
    neutral, _ = seqnet.loss_fn(logits, bnd, y, known, bt, seqnet.DEFAULTS,
                               team_weight=torch.ones((1, 4)), **args)
    assert torch.equal(base, neutral)


def test_team_disabled_windows_ignore_short_multipliers():
    torch = pytest.importorskip("torch")
    from mtah.models import seqnet
    logits = torch.tensor([[[0.2, -0.4, 0.8], [0.6, 0.1, -0.2]]])
    args = (logits, torch.zeros((1, 2)), torch.tensor([[H, A]]), torch.ones((1, 2), dtype=torch.bool),
            torch.zeros((1, 2)), seqnet.DEFAULTS)
    kw = dict(w=torch.ones(1), bnd_on=torch.tensor([False]), team_on=torch.zeros(1))
    a, _ = seqnet.loss_fn(*args, **kw)
    b, _ = seqnet.loss_fn(*args, team_weight=torch.full((1, 2), 2.), **kw)
    assert torch.equal(a, b)


def test_chunks_keep_neutral_padding_and_source_metadata():
    from mtah.models import seqnet
    y = np.array([H, A, A, D, H])
    seq = (np.zeros((5, 2), np.float32), y, np.ones(5, bool), np.arange(1, 6, dtype=np.float32), True, 0.5)
    weights = np.array([1, 2, 2, 1, 2], np.float32)
    p = dict(seqnet.DEFAULTS, chunk=8, chunk_step=4, bnd_sigma=0)
    chunk, = seqnet._chunks([seq], p, team_weights=[weights])
    assert np.array_equal(chunk[-1], [1, 2, 2, 1, 2, 1, 1, 1])
    assert np.array_equal(chunk[4], [1, 2, 3, 4, 5, 0, 0, 0])
    assert chunk[5] is True and chunk[6] == 0.5


def test_screening_configs_share_booster_cache_and_use_only_unseen_folds():
    from mtah import config
    from mtah.validation.splits import cache_tag
    ref = config.load("ev_short_base_rf0")
    trial = config.load("ev_short_team_rf0")
    source = config.load(ref["adapt"]["booster_source"])
    assert cache_tag(ref) == cache_tag(trial) == cache_tag(source)
    assert not set(ref["adapt"]["eval_matches"]) & set(ref["joint"]["matches"])
    assert ref["adapt"]["metrica_eval"] == trial["adapt"]["metrica_eval"] == [1, 2, 3]
    assert ref["seqnet"]["n_seeds"] == trial["seqnet"]["n_seeds"] == 2
    assert "short_team" not in ref["adapt"] and trial["adapt"]["short_team"]["factor"] == 2


def test_possession_diagnostics_measure_recall_and_false_splits():
    from mtah.validation.possession import diagnostics
    gt = np.array([H] * 10 + [A] * 10 + [H] * 40 + [D] * 10)
    pred = gt.copy()
    pred[15:20] = H
    pred[40:45] = A
    d = diagnostics(gt, pred, [(0, 70)], fps=10)
    assert d["n_gt"] == 3 and d["n_pred"] == 5
    assert d["duration_bins"]["1-2s"]["found"] == 2
    assert d["duration_bins"]["0-1s"]["n_pred"] == 2
    assert d["duration_bins"]["0-1s"]["matched_pred"] == 1


@pytest.mark.parametrize("mode", ["short", "repeat", "swap", "set"])
def test_screening_pipeline_keeps_external_test_matches_out_of_both_training_copies(tmp_path, monkeypatch, mode):
    """Exercise the real orchestration with synthetic data and a non-training model spy."""
    from types import SimpleNamespace
    import pandas as pd
    from mtah import config
    from mtah.models import seqnet
    from mtah.pipeline import adapt, metrica_src

    cfg = config.load("ev_set_pool_rf0" if mode == "set" else
                      ("ev_short_team_rf0" if mode == "short" else f"ev_ha_{mode}_rf0"))
    n, stride = 100, 5
    monkeypatch.setattr(adapt.C, "HALVES", {1: (0, n)})
    y = np.array([H] * 30 + [A] * 30 + [D] * 40)
    probs = np.full((n, 3), 1 / 3)
    feats = pd.DataFrame({"f": np.linspace(0, 1, n)})
    feats.attrs["group_columns"] = {"g": ["f"]}
    core = cfg["joint"]["matches"]
    matches = core + cfg["adapt"]["extra_matches"]
    indices = {m: i + 1 for i, m in enumerate(matches)}

    def prep(m, *_):
        f = feats.copy()
        f["f"] += indices.get(m, 99)
        data = SimpleNamespace(period=np.ones(n), labels=y, fps=25)
        return {"feats": f, "data": data, "corrupted": data, "aligned": y, "runs": [(0, n)]}

    monkeypatch.setattr(adapt, "OUTPUTS", tmp_path)
    monkeypatch.setattr(adapt.np, "load", lambda *_: probs.copy())
    monkeypatch.setattr(adapt, "common_groups", lambda _: ["g"])
    ctx = {"data": SimpleNamespace(), "feats": feats, "train_ok": np.ones(n, bool), "anchors": y}
    monkeypatch.setattr(adapt, "bar_context", lambda _: ctx)
    monkeypatch.setattr(adapt, "gap_lengths", lambda _: np.array([10]))
    monkeypatch.setattr(adapt, "bar_oof_probs", lambda *_: probs)
    bar_step = (np.zeros((n // stride, 4), np.float32), y[2::stride], np.ones(n // stride, bool), (0, n))
    monkeypatch.setattr(adapt, "anchor_steps", lambda *_: [bar_step])
    monkeypatch.setattr(adapt.decoder_tune, "lomo_probs", lambda *_: {m: probs for m in core})
    monkeypatch.setattr(adapt.decoder_tune, "unseen_probs", lambda _, ms, *a, **kw: {m: probs for m in ms})
    monkeypatch.setattr(adapt.decoder_tune, "eval_probs", lambda *_: {m: probs for m in matches})
    monkeypatch.setattr(adapt.dfl_referee, "prepare", prep)
    monkeypatch.setattr(metrica_src, "prepare", lambda g, _: prep(f"metrica_{g}"))
    monkeypatch.setattr(adapt, "suppress_short_returns", lambda labels, *_: labels)
    monkeypatch.setattr(adapt, "controlled_possession", lambda labels, *a, **kw: labels)
    # Preserve a source marker while using real step labels / slicing.
    original_step_view = adapt.step_view
    def step_view(f, *args):
        steps = original_step_view(f, *args)
        for x, _, _, _ in steps:
            x[:, 0] = float(f["f"].iloc[0])
        return steps
    monkeypatch.setattr(adapt, "step_view", step_view)
    if mode == "set":
        from mtah.models import player_set
        monkeypatch.setattr(player_set, "token_view", lambda data, cfg, runs, stride, p:
                            [np.zeros(((b-a)//stride, 216), np.float32) for a,b in runs])
    if mode in ("swap", "set"):
        from mtah import augmentation
        def swapped(prepared, cfg, columns, probs, targets, stride, **kw):
            result = []
            for x, y, known, run in step_view(prepared["feats"], columns, probs, targets, prepared["runs"], stride):
                x[:, 0] += 100
                if kw.get("token_params"):
                    x = np.c_[x, np.zeros((len(x), 216), np.float32)]
                result.append((x, augmentation.swap_labels(y), known, run))
            return result
        monkeypatch.setattr(augmentation, "swapped_steps", swapped)
    calls = []
    def train(seqs, p, seed, **kw):
        if mode == "set":
            assert p["token_shape"] == (24, 9)
            assert all(s[0].shape[1] == 220 for s in seqs)
        weights = kw["team_weights"]
        if mode == "short":
            assert len(weights) == len(seqs)
            assert all(len(w) == len(s[0]) for s, w in zip(seqs, weights))
        else:
            assert weights is None
        if "init" in kw and mode == "short":
            assert all((w == 1).all() for s, w in zip(seqs, weights) if s[0][0, 0] == 0)
        calls.append({"sources": {int(s[0][0, 0]) for s in seqs}, "seed": seed})
        return object()
    monkeypatch.setattr(seqnet, "train", train)
    def predict(models, steps, size, *_):
        if mode == "set":
            assert all(s[0].shape[1] == 220 for s in steps)
        return np.full((size, 3), 1 / 3)
    monkeypatch.setattr(adapt, "net_probs", predict)
    monkeypatch.setattr(adapt, "net_boundary", lambda models, steps, size, *_: np.zeros(size))
    monkeypatch.setattr(adapt.contact, "regime_for", lambda *_: None)
    monkeypatch.setattr(adapt.engine, "decode_runs", lambda *a, **kw: y)
    monkeypatch.setattr(adapt.m0, "decode_match", lambda *a, **kw: y)
    monkeypatch.setattr(adapt.dfl_referee, "score", lambda *_: {"spell_f1": 0.5, "macro_f1": 0.8})
    monkeypatch.setattr(adapt.pm, "evaluate", lambda *_: {"spell_f1_mean": 0.5, "macro_f1": 0.8})
    monkeypatch.setattr(adapt, "public_blocks", lambda: [])
    monkeypatch.setattr(adapt, "merged_gt", lambda _: y)
    monkeypatch.setattr(adapt, "official_reference", lambda *a, **kw: (y, {}, "synthetic"))
    monkeypatch.setattr(adapt, "reference_bands", lambda *_: {})
    monkeypatch.setattr(adapt, "_in_bands", lambda *_: (True, ""))
    monkeypatch.setattr(adapt, "score_blocks", lambda *a, **kw: {"ALL": {"spell_f1_mean": 0.5, "macro_f1": 0.8}})
    monkeypatch.setattr(adapt, "paired_spell_bootstrap", lambda *_: {"p_b_better": 0.5})
    monkeypatch.setattr(adapt.sanity, "stats", lambda _: {"dead_share": 0.4, "n_possessions": 2, "home_share": 0.5})
    monkeypatch.setattr(adapt.submission, "write", lambda path, _: path)
    monkeypatch.setattr(adapt.submission, "validate", lambda _: None)
    res = adapt.run(config="ev_short_team_rf0", name="synthetic", cfg=cfg)
    assert res["eval_matches"] == cfg["adapt"]["eval_matches"]
    assert len(calls) == 16  # three held-out folds plus all-data fit, two seeds, pretrain + adaptation
    for j, m in enumerate(cfg["adapt"]["eval_matches"]):
        assert all(indices[m] not in c["sources"] for c in calls[j * 4:(j + 1) * 4])
        assert all(indices[m] + 100 not in c["sources"] for c in calls[j * 4:(j + 1) * 4])
    if mode in ("swap", "set"):
        assert all(any(s > 100 for s in c["sources"]) for c in calls)
    assert all(set(indices.values()) <= c["sources"] for c in calls[-4:])
    assert all(len(r["per_match_gt35"]) == 3 for r in res["dfl_lomo"])
    assert all(len(r["per_match_gt35"]) == 3 for r in res["dfl_lomo_eval"])
    assert len(res["metrica_eval"]) == 3
