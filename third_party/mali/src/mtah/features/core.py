"""Compatibility shim: the M0 monolith now lives in features/groups/ (core + tags groups).

`from mtah.features.core import build_features` keeps working and builds the groups named in
the config (M0 = core + tags).
"""

from .registry import build_features  # noqa: F401
