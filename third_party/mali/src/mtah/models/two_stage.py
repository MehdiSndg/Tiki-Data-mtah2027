"""Two-stage frame model: P(alive) and P(home | alive), gradient boosting on CPU (deterministic).

    P(D) = 1 - p_alive,  P(H) = p_alive * p_home,  P(A) = p_alive * (1 - p_home)

model.family (Gün 2): "lgbm" (default, every earlier run) or "catboost" -- a second family on
the same features / weights / seeds for probability blending. CatBoost: symmetric (oblivious)
trees, ordered boosting; CPU with a fixed thread count and random_seed is deterministic.
"""

import lightgbm as lgb
import numpy as np

from ..constants import A, D, H


def _params(cfg, seed):
    p = dict(cfg["lgbm"])
    p.pop("row_stride", None)
    p.update(objective="binary", deterministic=True, force_col_wise=True, num_threads=8,
             random_state=seed, bagging_seed=seed, feature_fraction_seed=seed,
             data_random_seed=seed, verbose=-1)
    return p


CATBOOST_DEFAULTS = dict(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=5.0,
                         bootstrap_type="Bernoulli", subsample=0.7, rsm=0.6, border_count=128)


def _classifier(cfg, seed):
    family = cfg.get("model", {}).get("family", "lgbm")
    if family == "lgbm":
        return lgb.LGBMClassifier(**_params(cfg, seed))
    if family == "catboost":
        from catboost import CatBoostClassifier
        p = dict(CATBOOST_DEFAULTS, **cfg.get("catboost", {}))
        return CatBoostClassifier(**p, loss_function="Logloss", random_seed=seed, thread_count=8,
                                  verbose=False, allow_writing_files=False)
    raise ValueError(f"unknown model.family {family}")


def _gain(model, n_cols):
    if hasattr(model, "booster_"):
        return model.booster_.feature_importance("gain")
    return np.asarray(model.get_feature_importance(), float)[:n_cols]


class TwoStage:
    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, seed
        self.alive = None
        self.team = None
        self.columns = None

    def fit(self, X, y, team_ok=None, sample_weight=None):
        """X: DataFrame of anchor rows; y: H/A/D codes; team_ok: rows allowed in the team model;
        sample_weight: optional per-row weights (joint training balances sources with it)."""
        self.columns = list(X.columns)
        y = np.asarray(y)
        w = None if sample_weight is None else np.asarray(sample_weight, np.float64)
        if w is not None:
            w = w / max(float(np.mean(w)), 1e-12)        # mean 1: regularisation stays comparable
        self.alive = _classifier(self.cfg, self.seed)
        self.alive.fit(X, (y != D).astype(int), sample_weight=w)
        alive = np.isin(y, (H, A))
        if team_ok is not None:
            alive &= np.asarray(team_ok, bool)
        self.team = _classifier(self.cfg, self.seed + 1)
        w_team = None if w is None else w[alive] / max(float(np.mean(w[alive])), 1e-12)
        self.team.fit(X[alive], (y[alive] == H).astype(int), sample_weight=w_team)
        return self

    def predict(self, X):
        X = X[self.columns]
        return self.alive.predict_proba(X)[:, 1], self.team.predict_proba(X)[:, 1]

    def importance(self, top=15):
        out = {}
        for name, model in (("alive", self.alive), ("team", self.team)):
            gain = _gain(model, len(self.columns))
            order = np.argsort(gain)[::-1][:top]
            out[name] = [(self.columns[i], float(gain[i])) for i in order]
        return out


def to_probs(p_alive, p_home):
    """(T,3) float64 [H, A, D]."""
    p_alive = np.clip(np.asarray(p_alive, np.float64), 0, 1)
    p_home = np.clip(np.asarray(p_home, np.float64), 0, 1)
    return np.stack([p_alive * p_home, p_alive * (1 - p_home), 1 - p_alive], axis=1)
