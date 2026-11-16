"""Source-independent training / decoding engine shared by the BAR-ATH run, the DFL referee and
joint training.

Cost structure (user rules, 22 Oct):
  * the feature matrix of every source is built once and cached on disk per group;
  * ensemble member i uses LightGBM seed `seed + 1000 i` AND (masking.per_seed, default on) its
    own ball-masking draw `masking.seed + 1000 i`. Ball-free groups (geo, tags) of a masked copy
    are reused from the clean features; only ball-dependent groups are recomputed, and every
    masked copy is cached on disk under its own seed key, so a rerun only refits LightGBM;
  * seed probabilities are averaged before the HSMM decoder (seed ensemble); every seed is also
    decoded on its own so callers can see the run-to-run spread.
With n_seeds = 1 member 0 uses exactly the old fixed draw, so single-seed runs are unchanged.

Joint training: a list of TrainSource objects (BAR-ATH weak labels, DFL true labels, ...). Each
source's rows get a sample weight so that its TOTAL weight is its `share` of the training set,
whatever its row count.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import constants as C
from ..decoding.hsmm import decode_run, emission_logprob
from ..features import build_features
from ..labels.masking import mask_ball, mask_ball_by_state
from ..models.two_stage import TwoStage, to_probs


def lengths_tag(lengths):
    return hashlib.sha256(np.asarray(lengths, np.int64).tobytes()).hexdigest()[:8]


def n_seeds(cfg):
    return int(cfg["lgbm"].get("n_seeds", 1))


def model_seed(cfg, i):
    return int(cfg["seed"]) + 1000 * i


def mask_seed(cfg, i):
    mc = cfg["masking"]
    base = int(mc.get("seed", cfg["seed"]))
    return base + 1000 * i if mc.get("per_seed", False) else base


def masked_copies(data, cfg, tags, feats, base_key, lengths, seed=None, state_labels=None, feat_cfg=None):
    """[(features, newly_masked)] for cfg['masking']['copies'] masked copies of one draw.

    state_labels (true H/A/D codes, DFL only) switches to state-aware masking
    (masking.dead_fraction / masking.live_fraction); otherwise uniform target_fraction.
    feat_cfg: config the source's features were built with (DFL: common groups only)."""
    mc = cfg["masking"]
    seed = int(mc.get("seed", cfg["seed"])) if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(int(mc["copies"])):
        if state_labels is None:
            masked, newly = mask_ball(data, rng, mc["target_fraction"], lengths)
            key = f"{base_key}__mask_s{seed}_c{i}_f{mc['target_fraction']}_{lengths_tag(lengths)}"
        else:
            df, lf = mc.get("dead_fraction", 0.58), mc.get("live_fraction", 0.32)
            masked, newly = mask_ball_by_state(data, rng, state_labels, df, lf, lengths)
            key = f"{base_key}__smask_s{seed}_c{i}_d{df}_l{lf}_{lengths_tag(lengths)}"
        out.append((build_features(masked, feat_cfg or cfg, tags, reuse=feats, cache_key=key), newly))
    return out


def training_matrix(feats, copies, labels, team_ok, train_ok, stride):
    """(X, y, team_ok) from labelled frames of the match and the masked frames of its copies."""
    idx = np.flatnonzero(train_ok & (labels != C.X))[::stride]
    xs, ys, ts = [feats.iloc[idx]], [labels[idx]], [team_ok[idx]]
    for feats_m, newly in copies:
        idx_m = np.flatnonzero(train_ok & newly & (labels != C.X))[::stride]
        xs.append(feats_m.iloc[idx_m])
        ys.append(labels[idx_m])
        ts.append(team_ok[idx_m])
    return pd.concat(xs, ignore_index=True), np.concatenate(ys), np.concatenate(ts)


@dataclass
class TrainSource:
    """One block of training rows (a match + its masked copies) and its labels."""
    name: str
    data: object                      # TrackingData the copies are masked from
    feats: pd.DataFrame
    labels: np.ndarray
    team_ok: np.ndarray
    train_ok: np.ndarray
    cache_key: str
    lengths: np.ndarray
    tags: object = None
    stride: int = 2
    share: float = 1.0                # share of the total sample weight
    state_labels: np.ndarray = None   # true labels -> state-aware masking of the copies
    d_share: float = None             # weak sources: D rows carry this share of the source's
                                      # weight (keeps the anchor prior fixed when a new dead-ball
                                      # rule multiplies the D anchors)
    feat_cfg: dict = None             # feature config of this source (default: the run's cfg)
    _memo: dict = field(default_factory=dict, repr=False)

    def matrix(self, cfg, i):
        seed = mask_seed(cfg, i)
        if seed not in self._memo:
            self._memo.clear()        # keep one draw in memory at a time
            copies = masked_copies(self.data, cfg, self.tags, self.feats, self.cache_key,
                                   self.lengths, seed, self.state_labels, self.feat_cfg)
            self._memo[seed] = training_matrix(self.feats, copies, self.labels, self.team_ok,
                                               self.train_ok, self.stride)
        return self._memo[seed]


def _stack(sources, cfg, i, columns):
    xs, ys, ts, ws, counts = [], [], [], [], {}
    total_share = sum(s.share for s in sources)
    parts = [(s, *s.matrix(cfg, i)) for s in sources]
    n_total = sum(len(y) for _, _, y, _ in parts)
    for s, X, y, t in parts:
        xs.append(X.reindex(columns=columns))
        ys.append(y)
        ts.append(t)
        w = np.full(len(y), s.share / total_share * n_total / max(len(y), 1))
        if s.d_share is not None:
            is_d = y == C.D
            n_d, n_l = int(is_d.sum()), int((~is_d).sum())
            if n_d and n_l:
                w = w * np.where(is_d, s.d_share / n_d, (1 - s.d_share) / n_l) * len(y)
        ws.append(w)
        counts[s.name] = len(y)
    return (pd.concat(xs, ignore_index=True), np.concatenate(ys), np.concatenate(ts),
            np.concatenate(ws), counts)


def fit_predict_sources(sources, targets, cfg, columns=None):
    """Train the n-seed ensemble on `sources`; predict every DataFrame in `targets`.

    Returns (per_target: [[(p_alive, p_home)] per seed] per target, importance, row counts)."""
    columns = list(columns if columns is not None else sources[0].feats.columns)
    per_target = [[] for _ in targets]
    imp, counts = None, None
    weighted = len(sources) > 1 or any(s.d_share is not None for s in sources)
    for i in range(n_seeds(cfg)):
        X, y, t, w, counts = _stack(sources, cfg, i, columns)
        for fcfg in family_cfgs(cfg):
            model = TwoStage(fcfg, model_seed(cfg, i)).fit(X, y, t, sample_weight=w if weighted else None)
            for k, feats in enumerate(targets):
                per_target[k].append(model.predict(feats.reindex(columns=columns)))
            if imp is None:
                imp = model.importance(10)
    return per_target, imp, counts


def family_cfgs(cfg):
    """One config per model family of the ensemble. model.families (Gün 2 blending) trains every
    family on the same rows / weights / seed; members are averaged like seeds."""
    fams = cfg.get("model", {}).get("families")
    if not fams:
        return [cfg]
    out = []
    for f in fams:
        c = dict(cfg)
        c["model"] = dict(cfg.get("model", {}), family=f)
        out.append(c)
    return out


def fit_predict(X, y, t_ok, feats_pred, cfg):
    """Fixed-matrix variant: [(p_alive, p_home)] per seed and the first model's importance."""
    out, imp = [], None
    for i in range(n_seeds(cfg)):
        model = TwoStage(cfg, model_seed(cfg, i)).fit(X, y, t_ok)
        out.append(model.predict(feats_pred))
        if imp is None:
            imp = model.importance(10)
    return out, imp


def ensemble(per_seed):
    pa = np.mean([p for p, _ in per_seed], axis=0)
    ph = np.mean([h for _, h in per_seed], axis=0)
    return to_probs(pa, ph)


def decode_runs(probs, runs, cfg, forced_dead=(), n=None, boundary=None, contested=None, regime=None):
    """Labels decoded inside each run, X elsewhere. forced_dead: [(start, stop)] spans.
    boundary: optional per-frame boundary probability (same length as probs)."""
    n = n or len(probs)
    lp = emission_logprob(probs, cfg)
    labels = np.full(n, C.X, np.int8)
    for a, b in runs:
        forced = np.zeros(b - a, bool)
        for fa, fb in forced_dead:
            lo, hi = max(a, fa), min(b, fb)
            if lo < hi:
                forced[lo - a:hi - a] = True
        labels[a:b] = decode_run(lp[a:b], cfg, forced, None if boundary is None else boundary[a:b],
                                 None if contested is None else contested[a:b],
                                 None if regime is None else {k: (None if v is None else v[a:b]) for k, v in regime.items()})
    return labels
