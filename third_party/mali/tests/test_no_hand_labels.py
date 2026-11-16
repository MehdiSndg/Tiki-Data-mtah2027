"""Upstream rule change (RULES c073f29 / f8f2629): no hand labels on the provided game, in any
amount and for any purpose. The submission pipeline must not read labels/ at all."""
import re
from pathlib import Path

import numpy as np
import pytest

from mtah.validation.splits import (Block, cache_tag, labelled_blocks, public_blocks, trains_on_blocks,
                                    training_blocks, training_mask)

ROOT = Path(__file__).resolve().parents[1]
CODE = [ROOT / "run.py", *sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "tools").glob("*.py"))]
FORBIDDEN = [r"LABELS_DIR", r"hand_labels\s*\(", r"labels_fingerprint", r"/\s*[\"']labels[\"']",
             r"labels\W+\*?\w*\.csv", r"\blobo\b", r"cv_fold_blocks", r"annotator", r"local_validator"]


@pytest.mark.parametrize("path", CODE, ids=lambda p: str(p.relative_to(ROOT)))
def test_code_never_touches_hand_labels(path):
    text = path.read_text(encoding="utf-8")
    for pat in FORBIDDEN:
        assert not re.search(pat, text), f"{path}: matches {pat}"


def test_only_public_window_is_labelled(tmp_path, monkeypatch):
    # a CSV in a labels/ folder next to the code changes nothing
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "x.csv").write_text("frame,label\n20000,H\n20001,H\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert [b.name for b, _ in labelled_blocks()] == ["public"]
    assert [b.name for b, _ in public_blocks()] == ["public"]
    assert labelled_blocks(include_public=False) == []


def test_hand_block_cross_fitting_is_retired():
    with pytest.raises(ValueError):
        training_blocks({"validation": {"cv": [0, 2]}}, [])


def test_holdout_semantics():
    blocks = [(Block("public", 11185, 18685), None)]
    sub = {"validation": {"holdout": "none"}}
    assert trains_on_blocks(sub) and training_blocks(sub, blocks) == []
    m = training_mask(training_blocks({"validation": {}}, blocks), 1125)
    assert not m[11185 - 1125:18685 + 1125].any() and m[:11185 - 1125].all()


def test_main_config_is_region_fold_only():
    from mtah import config
    cfg = config.load("main")
    v = cfg["validation"]
    assert v.get("region_folds") and not v.get("cv")
    for k in range(int(v["region_folds"])):
        fv = config.load(f"main_rf{k}")["validation"]
        assert list(fv["region_cv"]) == [k, int(v["region_folds"])] and not fv.get("cv")


def test_cache_tag_follows_booster_config_not_net_config():
    from mtah import config
    a = config.load("main")
    b = config.load("main")
    b["features"] = dict(b["features"], groups=list(b["features"]["groups"]) + ["ck"])
    c = config.load("main")
    c["seqnet"] = dict(c["seqnet"], n_seeds=1)
    assert cache_tag(a) != cache_tag(b) and cache_tag(a) == cache_tag(c)
    assert re.fullmatch(r"_c[0-9a-f]{8}", cache_tag(a))


def test_no_source_file_is_gitignored():
    """25 Oct: an unanchored 'labels/' pattern silently dropped src/mtah/labels/ from the repo."""
    import shutil
    import subprocess
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    files = [str(p.relative_to(ROOT)) for p in ROOT.joinpath("src").rglob("*.py") if "__pycache__" not in p.parts]
    files += [str(p.relative_to(ROOT)) for p in (ROOT / "configs").glob("*.yaml")]
    r = subprocess.run(["git", "check-ignore", "--no-index", *files], cwd=ROOT, capture_output=True, text=True)
    assert r.stdout.strip() == "", f"ignored by .gitignore: {r.stdout}"
