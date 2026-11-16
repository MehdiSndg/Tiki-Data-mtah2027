"""Validation blocks and leakage-safe training masks.

Upstream rule change (RULES c073f29 / f8f2629): no hand labels on the provided game, in any amount
and for any purpose. The only labelled span of BAR-ATH is therefore the organisers' public
5 minutes, used as a format / sanity diagnostic only (never for training or selection). Nothing in
this package reads labels/; region folds (validation.regions) are label-independent.
Training never uses frames inside a held-out block or within `buffer` frames of it.
"""

from dataclasses import dataclass

import numpy as np

from ..constants import N_FRAMES, PUBLIC_WINDOW, X
from ..io import gt as gt_io


@dataclass(frozen=True)
class Block:
    name: str
    start: int
    stop: int        # half-open

    @property
    def n_frames(self):
        return self.stop - self.start


def labelled_blocks(include_public=True):
    """[(Block, codes)]: the organisers' public window only (hand labels are retired)."""
    return [(Block("public", *PUBLIC_WINDOW), gt_io.public_gt())] if include_public else []


def public_blocks():
    """The organisers' public window (diagnostic score only)."""
    return labelled_blocks(include_public=True)


def merged_gt(blocks):
    """One full-length GT array holding the labels of the given blocks (X elsewhere).

    X between blocks makes the official scorer treat every block as its own graded run, so the
    unmodified metric scores exactly the labelled spans.
    """
    gt = np.full(N_FRAMES, X, np.int8)
    for block, codes in blocks:
        gt[block.start:block.stop] = codes[block.start:block.stop]
    return gt


def trains_on_blocks(cfg):
    """True when `validation.holdout: none` and no region fold: weak anchors of every frame train,
    the public window's frames included (its GT never trains)."""
    v = cfg.get("validation") or {}
    return v.get("holdout", "all") == "none" and not v.get("region_cv")


def training_blocks(cfg, blocks=None):
    """Blocks whose weak anchors a model of this config must not train on.

    `validation.region_cv: [k, n]` (the submission's region folds): the label-independent chunks
    of region k. Otherwise the public window, unless `holdout: none`. Hand-block cross-fitting
    (`validation.cv`) is retired with the hand labels."""
    v = cfg.get("validation") or {}
    if v.get("cv"):
        raise ValueError("validation.cv (hand-block cross-fitting) is retired: hand labels may not be used")
    rcv = v.get("region_cv")
    if rcv:
        from .regions import region_chunks
        k = int(rcv[0])
        return [(Block(f"region{k}_{i}", s, e), None) for i, (s, e, r) in enumerate(region_chunks(cfg)) if r == k]
    if trains_on_blocks(cfg):
        return []
    return labelled_blocks() if blocks is None else blocks


def training_mask(blocks, buffer, n=N_FRAMES):
    """True where a frame may be used for training (outside every block +- buffer)."""
    mask = np.ones(n, dtype=bool)
    for block, _ in blocks:
        mask[max(0, block.start - buffer):min(n, block.stop + buffer)] = False
    return mask


BOOSTER_KEYS = ("seed", "validation", "preprocess", "features", "masking", "dfl_referee", "weak_labels",
                "lgbm", "joint")


def cache_tag(cfg):
    """File-name tag of caches produced by a config's booster stage (DFL LOMO / unseen / BAR OOF
    probabilities): a digest of every config section the booster depends on (no label state)."""
    from .. import config as config_mod
    part = {k: cfg.get(k) for k in BOOSTER_KEYS}
    if cfg.get("dfl_mix_eval"):              # Gün 7: mixed-corruption booster (only when set: other tags unchanged)
        part["dfl_mix_eval"] = cfg.get("dfl_referee_eval")
    if cfg.get("dfl_referee_train"):         # 5 Nov: v3_dr training corruption (only when set)
        part["dfl_referee_train"] = cfg.get("dfl_referee_train")
    if cfg.get("metrica_labels") and (cfg.get("joint") or {}).get("metrica_booster"):   # 11 Nov: ball-status labels
        part["metrica_labels"] = [cfg.get("metrica_labels"), cfg.get("metrica_min_dead", 50)]
        if cfg.get("metrica_dead_fill"):
            part["metrica_dead_fill"] = True
        if cfg.get("metrica_dead_drift"):
            part["metrica_dead_drift"] = cfg.get("metrica_dead_drift")
    return f"_c{config_mod.digest(part)[:8]}"
