"""The official metric library, imported unmodified from the competition repository."""

import sys

from ..paths import OFFICIAL

_EVAL = str(OFFICIAL / "evaluation")
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import possession_metrics as pm  # noqa: E402,F401
