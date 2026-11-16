"""Cloud mode (5 Nov 2026): train on a rented GPU WITHOUT any data of the provided game.

MTAH_CLOUD=1 makes the external-data-only pipeline run with no BAR-ATH files present:
- the only BAR-derived input is the list of ball-gap lengths that the DFL ball masking samples from
  (an aggregate statistic, exported locally by `export_bar_lengths`, a few KB, under
  CACHE/cloud/bar_gap_lengths.npy);
- the booster stage only reports its feature columns (taken from an external match; the booster folds
  themselves are trained on DFL by joint.evaluate_lomo / unseen_probs as usual);
- `adapt` trains and scores on external data, saves the final nets (outputs/<run>/nets/) and skips the
  provided game; tools/bar_from_nets.py then predicts and checks the provided game locally.
Only external_only configs are allowed. Nothing of the provided game is uploaded.
"""

import os

import numpy as np

from .paths import CACHE, ensure

LENGTHS = "bar_gap_lengths.npy"


def enabled():
    return os.environ.get("MTAH_CLOUD") == "1"


def lengths_path():
    return CACHE / "cloud" / LENGTHS


def bar_lengths(cfg):
    """Ball-gap lengths of the provided game's tracking: from the exported file in cloud mode."""
    if enabled():
        path = lengths_path()
        if not path.is_file():
            raise FileNotFoundError(f"cloud mode needs {path} (tools/cloud_export.py writes it locally)")
        return np.load(path)
    from .io.broadcast import bar_data
    from .labels.masking import gap_lengths
    return gap_lengths(bar_data(cfg))


def export_bar_lengths(cfg):
    from .io.broadcast import bar_data
    from .labels.masking import gap_lengths
    lengths = gap_lengths(bar_data(cfg)).astype(np.int64)
    path = ensure(lengths_path().parent) / LENGTHS
    np.save(path, lengths)
    return path, lengths


def require_external(cfg):
    if enabled() and not cfg.get("joint", {}).get("external_only"):
        raise ValueError("cloud mode runs external_only configs only")


def external_columns(cfg, lengths):
    """Feature columns of the source-common groups, read from the first joint DFL match."""
    from .pipeline import dfl_referee, joint
    m = joint.available(cfg["joint"]["matches"])[0]
    return list(dfl_referee.prepare(m, cfg, lengths)["feats"].columns)
