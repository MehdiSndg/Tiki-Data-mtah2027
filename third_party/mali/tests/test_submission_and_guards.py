"""Submission contract and leakage guards."""

from pathlib import Path

import numpy as np
import pytest

from mtah import constants as C
from mtah.io import submission
from mtah.validation.splits import Block, training_mask

SRC = Path(__file__).resolve().parents[1] / "src" / "mtah"


def test_writer_produces_valid_lf_file(tmp_path):
    codes = np.full(C.N_FRAMES, C.X, np.int8)
    codes[C.graded_mask()] = C.D
    path = submission.write(tmp_path / "s.csv", codes)
    assert submission.validate(path) is None
    raw = path.read_bytes()
    assert b"\r\n" not in raw and raw.startswith(b"frame,label\n")


def test_writer_refuses_x_on_graded_frames(tmp_path):
    codes = np.full(C.N_FRAMES, C.X, np.int8)
    with pytest.raises(ValueError):
        submission.write(tmp_path / "s.csv", codes)


def test_training_mask_respects_buffer():
    block = Block("b", 20_000, 21_000)
    mask = training_mask([(block, None)], buffer=1125)
    assert not mask[20_000 - 1125] and not mask[21_000 + 1124]
    assert mask[20_000 - 1126] and mask[21_000 + 1125]


@pytest.mark.parametrize("folder", ["features", "models", "labels", "decoding", "preprocess.py"])
def test_feature_and_model_code_never_reads_ground_truth(folder):
    paths = [SRC / folder] if folder.endswith(".py") else list((SRC / folder).glob("*.py"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "io.gt" not in text and "io import gt" not in text and "public_gt" not in text, path
