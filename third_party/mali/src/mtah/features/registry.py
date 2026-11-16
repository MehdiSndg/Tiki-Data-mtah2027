"""Feature groups and the single entry point that assembles them.

A group is a function ctx -> dict[column -> array] plus metadata:
    source_specific  True when the inputs exist for one source only (e.g. Smart Tagging).
                     Joint training across sources must leave such groups out.
    ball_free        True when no column depends on the ball (lets masked copies reuse it).
    context          columns that get centred rolling means / future-past contrasts.
"""

import hashlib
import inspect
import json
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..paths import CACHE
from .context import FeatureContext


@dataclass
class Group:
    name: str
    fn: callable
    source_specific: bool = False
    ball_free: bool = False
    context: list = field(default_factory=list)


GROUPS = {}


def register(name, source_specific=False, ball_free=False, context=()):
    def wrap(fn):
        GROUPS[name] = Group(name, fn, source_specific, ball_free, list(context))
        return fn
    return wrap


def _context(df, signals, windows, contrast):
    out = {}
    for name in signals:
        s = df[name].astype("float64")
        rev = s.iloc[::-1]
        for w in windows:
            out[f"{name}_m{w}"] = s.rolling(2 * w + 1, center=True, min_periods=1).mean()
        for w in contrast:
            past = s.rolling(w, min_periods=1).mean().shift(1)
            fut = rev.rolling(w, min_periods=1).mean().shift(1).iloc[::-1]
            out[f"{name}_fb{w}"] = fut - past
    return pd.DataFrame(out, index=df.index)


def group_hash(name, cfg):
    """Fingerprint of everything a group's output depends on: its code, the shared context /
    registry code, and the feature + preprocessing config. Any change invalidates the cache."""
    group = GROUPS[name]
    module = sys.modules[group.fn.__module__]
    deps = [inspect.getsource(sys.modules[m]) for m in getattr(module, "DEPENDS", ())]
    parts = [name,
             inspect.getsource(module), *deps,
             inspect.getsource(sys.modules[FeatureContext.__module__]),
             inspect.getsource(_context),
             json.dumps(cfg["features"], sort_keys=True),
             json.dumps(cfg["preprocess"], sort_keys=True)]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def _compute_group(group, ctx, fc):
    cols = {k: np.asarray(v, np.float32) for k, v in group.fn(ctx).items()}
    df = pd.DataFrame(cols)
    if group.context:
        df = pd.concat([df, _context(df, group.context, fc["context_windows"],
                                     fc["contrast_windows"]).astype(np.float32)], axis=1)
    return df


def build_features(data, cfg, tags=None, groups=None, reuse=None, cache_key=None):
    """DataFrame of the requested groups (default: cfg['features']['groups']).

    tags: Smart Tagging table for sources that have one. reuse: a previously built frame for the
    same players (e.g. the unmasked match) whose ball-free groups are copied instead of recomputed.
    cache_key: if given, each group is read from / written to
    cache/features/<cache_key>/<group>_<hash>.parquet. The key must identify the exact input
    data (source + any masking); the hash covers code and config.
    """
    from . import groups as _groups  # noqa: F401  (registers every group)

    names = list(groups or cfg["features"]["groups"])
    ctx = FeatureContext(data, cfg)
    ctx.tags = tags
    fc = cfg["features"]
    parts = []
    for name in names:
        group = GROUPS[name]
        if group.source_specific and tags is None:
            raise ValueError(f"group '{name}' needs source-specific inputs this source lacks")
        if reuse is not None and group.ball_free and set(reuse.attrs.get("groups", [])) >= {name}:
            cols = reuse.attrs["group_columns"][name]
            parts.append(reuse[cols])
            continue
        path = None
        if cache_key is not None:
            path = CACHE / "features" / cache_key / f"{name}_{group_hash(name, cfg)}.parquet"
            if path.is_file():
                parts.append(pd.read_parquet(path))
                continue
        df = _compute_group(group, ctx, fc)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
        parts.append(df)
    out = pd.concat(parts, axis=1)
    dup = out.columns[out.columns.duplicated()].tolist()
    if dup:
        raise ValueError(f"duplicate feature columns across groups: {dup}")
    out.attrs["groups"] = names
    out.attrs["group_columns"] = {n: list(p.columns) for n, p in zip(names, parts)}
    out.attrs["gk_dir"] = ctx.gk_dir
    return out


def common_groups(names):
    """The subset of groups usable for joint training across sources."""
    from . import groups as _groups  # noqa: F401
    return [n for n in names if not GROUPS[n].source_specific]
