"""Cloud mode: no provided-game data; nets saved and rebuilt for local prediction."""

import sys
from pathlib import Path

import numpy as np
import pytest

from mtah import cloud


def test_cloud_mode_reads_lengths_from_file_and_refuses_non_external(monkeypatch, tmp_path):
    monkeypatch.setenv("MTAH_CLOUD", "1")
    f = tmp_path / "bar_gap_lengths.npy"
    np.save(f, np.array([3, 40, 250]))
    monkeypatch.setattr(cloud, "lengths_path", lambda: f)
    assert list(cloud.bar_lengths({})) == [3, 40, 250]
    with pytest.raises(ValueError):
        cloud.require_external({"joint": {"external_only": False}})
    cloud.require_external({"joint": {"external_only": True}})
    from mtah.pipeline import m0
    with pytest.raises(ValueError):
        m0.run(cfg={"joint": {"external_only": False}, "name": "x"})


def test_cloud_mode_is_off_by_default(monkeypatch):
    monkeypatch.delenv("MTAH_CLOUD", raising=False)
    assert not cloud.enabled()


def test_saved_nets_rebuild_identical_predictions(tmp_path):
    torch = pytest.importorskip("torch")
    from mtah.models import seqnet
    from mtah.pipeline import adapt
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import bar_from_nets
    p = dict(seqnet.DEFAULTS, hidden=16, gru_hidden=8, gru_layers=1, dilations=(1, 2))
    torch.manual_seed(0)
    nets = [seqnet.SeqNet(5, p).eval(), seqnet.SeqNet(5, p).eval()]
    adapt.save_nets(tmp_path, nets, [2027, 3027], p, 5, ["a", "b"])
    loaded, columns = bar_from_nets.load_nets(tmp_path)
    assert [s for s, _ in loaded] == [2027, 3027] and columns == ["a", "b"]
    x = np.random.default_rng(0).normal(size=(40, 5)).astype(np.float32)
    for (_, b), a in zip(loaded, nets):
        a = a.to(seqnet.device())
        np.testing.assert_array_equal(seqnet.predict(a, x), seqnet.predict(b, x))
