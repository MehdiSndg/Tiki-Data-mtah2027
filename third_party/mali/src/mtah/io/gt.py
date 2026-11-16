"""The organisers' public ground truth (5-minute window). Only the validation layer may import
this module. It is a format / sanity diagnostic, never training data or a tuning set.

Hand labels on BAR-ATH are retired (upstream rule change: no hand labels on the provided game, in
any amount and for any purpose); nothing here reads labels/.
"""

import numpy as np
import pandas as pd

from ..constants import LABELS, N_FRAMES
from ..paths import PUBLIC_GT


def codes_from_labels(values):
    values = np.array([str(v).strip().upper() for v in values], dtype=object)
    codes = np.full(len(values), -1, np.int8)
    for i, name in enumerate(LABELS):
        codes[values == name] = i
    if (codes < 0).any():
        raise ValueError(f"unknown labels: {sorted(set(values[codes < 0]))}")
    return codes


def public_gt():
    """Full-length codes of the public ground truth (X outside the 5-minute window).
    Tiki-Data working copy: with MTAH_NO_PUBLIC set the file is never read; a placeholder (D inside the window, X
    outside) is returned, so the diagnostic public scores printed by m0 / adapt are meaningless and nothing else
    depends on it in an external_only run (training uses only external sources there)."""
    import os
    if os.environ.get("MTAH_NO_PUBLIC"):
        from ..constants import PUBLIC_WINDOW
        codes = np.full(N_FRAMES, LABELS.index("X"), np.int8)
        codes[PUBLIC_WINDOW[0]:PUBLIC_WINDOW[1]] = LABELS.index("D")
        return codes
    table = pd.read_csv(PUBLIC_GT, dtype={"label": "object"})
    codes = codes_from_labels(table["label"].to_numpy())
    if len(codes) != N_FRAMES:
        raise ValueError("public GT has the wrong length")
    return codes
