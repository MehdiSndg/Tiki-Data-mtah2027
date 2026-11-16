"""Training-only external team relabelling; raw coordinates are never mirrored."""

from dataclasses import replace
import hashlib
import inspect

import numpy as np

from . import constants as C


def swap_labels(labels):
    y = np.asarray(labels)
    return np.where((y == C.H) | (y == C.A), 1 - y, y).astype(y.dtype)


def swap_tracking(data):
    """Share unchanged coordinates; exchange slot teams and targets without changing source identity.

    FeatureContext infers goalkeeper slots and attack direction again from the exchanged teams.
    Keeping the source name is intentional: the twin must stay in the same outer match fold.
    """
    if not np.isin(data.team, [C.H, C.A]).all():
        raise ValueError("Team exchange requires binary team assignments")
    meta = dict(data.meta)
    if "home" in meta and "away" in meta:
        meta["home"], meta["away"] = meta["away"], meta["home"]
    return replace(data, team=(1 - data.team).astype(data.team.dtype), meta=meta,
                   labels=None if data.labels is None else swap_labels(data.labels))


def swapped_steps(prep, cfg, columns, probs, targets, stride, token_params=None):
    """Recompute every common feature from an already-corrupted external track.

    Corruption is not drawn again using the exchanged labels. The frozen booster is not refit:
    its H/A input probabilities are permuted as part of the synthetic training example.
    Real held-out/inference features and booster predictions remain unchanged.
    """
    from .pipeline.hybrid import step_view

    data, feats = swapped_features(prep, cfg, columns)
    steps = step_view(feats, columns, np.asarray(probs)[:, SWAP], None if targets is None else swap_labels(targets),
                      prep["runs"], stride)
    if token_params is not None:
        from .models.player_set import token_view
        from .pipeline.hybrid import with_tokens
        steps = with_tokens(steps, token_view(data, cfg, prep["runs"], stride, token_params))
    return steps


SWAP = [C.A, C.H, C.D]          # an involution: the [H, A, D] columns of the team-exchanged view


def swapped_features(prep, cfg, columns):
    """Team-exchanged tracking and its common features (cached), checked against the original."""
    from .features import build_features, common_groups

    data = swap_tracking(prep["corrupted"])
    version = hashlib.sha256(inspect.getsource(swap_tracking).encode()).hexdigest()[:10]
    feats = build_features(data, cfg, groups=common_groups(cfg["features"]["groups"]),
                           cache_key=prep["key"] + "__team_exchange_" + version)
    if list(feats.columns) != list(columns):
        raise ValueError("Exchanged feature columns differ from the reference")
    for period, (home_gk, away_gk, positive) in prep["feats"].attrs.get("gk_dir", {}).items():
        if feats.attrs["gk_dir"][period] != (away_gk, home_gk, not positive):
            raise ValueError("Cannot exchange a source with ambiguous goalkeeper/attack direction")
    return data, feats


def tta_probs(p_orig, p_swap):
    """Test-time team-exchange average (Rapor 9 Sıra 1). p_swap is the net's output on the exchanged
    view, still in that view's [H, A, D] order; it is mapped back and geometrically averaged with
    p_orig before any decoding (never label voting after the decoder)."""
    lp = 0.5 * (np.log(np.clip(p_orig, 1e-6, 1.0)) + np.log(np.clip(np.asarray(p_swap)[:, SWAP], 1e-6, 1.0)))
    q = np.exp(lp - lp.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True)


def tta_boundary(b_orig, b_swap):
    """Boundary heads are team-agnostic: plain mean (0 outside runs stays 0), decoder grid rounding."""
    return np.round(0.5 * (np.asarray(b_orig) + np.asarray(b_swap)), 6)


def training_steps(matches, primary, mixed, copies=None, exclude=None):
    """One ordering for real/corrupted/twin examples; exclude the source and ALL its twins."""
    result = ([s for m in matches if m != exclude for s in primary[m]]
              + [s for m in mixed if m != exclude for s in mixed[m]])
    if copies:
        result += [s for m in copies if m != exclude for s in copies[m]]
    return result
