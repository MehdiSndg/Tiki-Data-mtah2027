"""YAML configs: base.yaml merged with an experiment file."""

import copy
import hashlib
import json

import yaml

from .paths import CONFIGS


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read(name):
    with open(CONFIGS / f"{name}.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load(name="m0"):
    """base.yaml <- [inherited experiment chain] <- configs/<name>.yaml.

    An experiment file may say `inherit: m0` to start from another experiment.
    """
    chain, current = [], name
    while current:
        exp = _read(current)
        chain.append(exp)
        current = exp.pop("inherit", None)
        if len(chain) > 20:
            raise ValueError("config inheritance loop")
    cfg = _read("base")
    for exp in reversed(chain):
        cfg = _merge(cfg, exp)
    return cfg


def with_groups(cfg, groups, name=None):
    """Copy of cfg using a different feature-group list (used by the ablation driver)."""
    out = copy.deepcopy(cfg)
    out["features"]["groups"] = list(groups)
    if name:
        out["name"] = name
    return out


def digest(cfg):
    """Short stable hash of a config, written next to every result."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
