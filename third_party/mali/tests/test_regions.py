"""Gün 5 (Ek N.6): region folds of the submission model."""
import numpy as np
import pytest

from mtah import constants as C
from mtah.paths import CACHE
from mtah.validation.splits import trains_on_blocks, training_blocks, training_mask

needs_bar = pytest.mark.skipif(not (CACHE / "barath").exists(), reason="needs cache/barath")


@needs_bar
def test_region_chunks_tile_the_graded_halves_and_alternate():
    from mtah.validation.regions import region_chunks
    cfg = {"validation": {"region_folds": 2}}
    ch = region_chunks(cfg)
    assert len(ch) == 8 and [r for _, _, r in ch] == [0, 1] * 4
    for (a, b), half in zip([(ch[0][0], ch[3][1]), (ch[4][0], ch[7][1])], C.HALVES.values()):
        assert (a, b) == half
    for (s0, e0, _), (s1, _, _) in zip(ch[:-1], ch[1:]):
        assert e0 == s1 or e0 in (C.HALVES[1][1],)


@needs_bar
def test_region_folds_hold_out_complementary_regions_and_ignore_labels(monkeypatch):
    k0 = {"validation": {"region_folds": 2, "region_cv": [0, 2]}}
    k1 = {"validation": {"region_folds": 2, "region_cv": [1, 2]}}
    m0, m1 = training_mask(training_blocks(k0, []), 1125), training_mask(training_blocks(k1, []), 1125)
    graded = np.zeros(C.N_FRAMES, bool)
    for a, b in C.HALVES.values():
        graded[a:b] = True
    assert not (m0 & m1 & graded).any()            # no graded frame trains both folds
    assert (m0 | m1)[graded].mean() > 0.85        # buffers only cost a little
    assert not trains_on_blocks(k0)
    # the cut points never depend on a block list passed in (label-independent)
    assert [b.start for b, _ in training_blocks(k0, [])] == [b.start for b, _ in training_blocks(k0, None)]


@needs_bar
def test_stitch_takes_each_region_from_its_own_fold(tmp_path, monkeypatch):
    from mtah import config
    from mtah.pipeline import regionrun
    from mtah.validation.regions import region_of_frames
    monkeypatch.setattr(regionrun, "OUTPUTS", tmp_path)
    cfg = config.load("main")
    cfg["adapt"]["decoder"]["contact"] = {"gate": False}
    reg = region_of_frames(cfg, C.N_FRAMES)
    for k in (0, 1):
        d = tmp_path / f"f{k}"
        d.mkdir()
        p = np.tile([0.9, 0.09, 0.01] if k == 0 else [0.09, 0.9, 0.01], (C.N_FRAMES, 1))
        np.save(d / "probs.npy", p)
        np.save(d / "bar_bnd_adapted.npy", np.zeros(C.N_FRAMES))
    lab, s = regionrun.stitch(cfg, [["f0", "f1"]], "out")
    g = np.zeros(C.N_FRAMES, bool)
    for a, b in C.HALVES.values():
        g[a + 2000:b - 2000] = True
    inner = g & (lab != C.D)
    assert (lab[inner & (reg == 0)] == C.H).mean() > 0.95 and (lab[inner & (reg == 1)] == C.A).mean() > 0.95
    assert s["official_format_check"] == "OK" and s["hand_labels"] == "none"


def test_temperature_match_restores_member_entropy():
    from mtah.pipeline.regionrun import _entropy, temperature_match
    rng = np.random.default_rng(0)
    a = rng.dirichlet([0.3, 0.3, 0.3], 5000)
    b = rng.dirichlet([0.3, 0.3, 0.3], 5000)
    ens = (a + b) / 2
    mask = np.ones(5000, bool)
    out, tau = temperature_match(ens, [a, b], mask)
    target = (_entropy(a).mean() + _entropy(b).mean()) / 2
    assert tau < 1.0 and abs(_entropy(out).mean() - target) < 0.01
    assert np.allclose(out.sum(1), 1.0, atol=1e-5)
    same, t1 = temperature_match(a, [a], mask)
    assert t1 == 1.0 and same is a
