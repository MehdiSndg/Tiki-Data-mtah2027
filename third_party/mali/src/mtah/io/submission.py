"""Write and check the submission file exactly as the official scorer expects it."""

from pathlib import Path

import numpy as np

from ..constants import LABELS, N_FRAMES, X, graded_mask
from ..validation.official import pm


def write(path, codes):
    """frame,label CSV with '\\n' line endings (byte-identical on every OS). Returns the path."""
    codes = np.asarray(codes, np.int8)
    if len(codes) != N_FRAMES:
        raise ValueError(f"{len(codes)} labels, expected {N_FRAMES}")
    n_x = int(np.sum(codes[graded_mask()] == X))
    if n_x:
        raise ValueError(f"{n_x} graded frames are X; forward-fill them before writing")
    letters = np.array(LABELS)[codes]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{i},{lab}\n" for i, lab in enumerate(letters))
    return path


def validate(path):
    """None if the file passes the official checks, else the official one-line reason."""
    table = pm.read_label_csv(str(path))
    return pm.validate_submission(table, N_FRAMES)
