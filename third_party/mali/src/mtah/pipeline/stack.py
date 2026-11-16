"""Calibration + hierarchical stacking of the boosting and temporal-net probabilities
(autonomous loop, 22 Oct; report 2 section 9.1, report 3 section 7.1).

Level 1  vector scaling per model: p = softmax(log p / T + b), b = (0, b_A, b_D), fitted by NLL
         (scipy L-BFGS-B, deterministic) -- isotonic regression is NOT used (report 3: step
         artefacts create fake transitions); then log-pool  w * log p_boost + (1 - w) * log p_net.
Level 2  meta-refiner: multinomial L2 logistic regression on [calibrated log-probs of both
         models, their centred rolling means over 1 s and 5 s, both entropies, ball visibility and
         its interactions]. Target: the DFL labels aligned to Metrica semantics (35 frames +
         control) or the training alignment (11 frames).
Level 2b optional Saerens-Latinne-Decaestecker EM prior-shift correction per match (uses only
         the predictions of that match: no labels).
Level 3  the config's HSMM (unchanged).
Everything is fitted on DFL with leave-one-match-out nesting (the held-out match is never used
for fitting its own calibration / stacker); the BAR-ATH version is fitted on all DFL matches.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .. import constants as C

EPS = 1e-6


def _logp(p):
    return np.log(np.clip(p, EPS, 1.0))


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def vector_scale_fit(p, y):
    lp = _logp(p)

    def nll(theta):
        T = np.exp(theta[0])
        z = lp / T + np.array([0.0, theta[1], theta[2]])
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        return float(np.mean(lse - z[np.arange(len(y)), y]))

    res = minimize(nll, np.zeros(3), method="L-BFGS-B")
    return res.x


def vector_scale_apply(p, theta):
    return _softmax(_logp(p) / np.exp(theta[0]) + np.array([0.0, theta[1], theta[2]]))


def log_pool(pb, pn, w):
    return _softmax(w * _logp(pb) + (1 - w) * _logp(pn))


def meta_features(pb, pn, vis, extras=()):
    lb, ln = _logp(pb), _logp(pn)
    lx = [_logp(e) for e in extras]
    cols = [lb, ln, *lx]
    for win in (25, 125):
        for arr in (lb, ln, *lx):
            cols.append(pd.DataFrame(arr).rolling(win, center=True, min_periods=1).mean().to_numpy())
    ent = [-(p * _logp(p)).sum(1, keepdims=True) for p in (pb, pn, *extras)]
    v = vis.astype(np.float64)[:, None]
    cols += ent + [v, v * ln[:, 2:3], v * lb[:, 2:3]]
    return np.concatenate(cols, axis=1)


def em_prior_shift(p, prior_train, iters=50, tol=1e-6):
    """Saerens et al. (2002): re-estimate the class priors of an unlabelled set and adjust."""
    prior = prior_train.copy()
    adj = p
    for _ in range(iters):
        adj = p * (prior / prior_train)
        adj = adj / adj.sum(1, keepdims=True)
        new = adj.mean(0)
        if np.abs(new - prior).max() < tol:
            prior = new
            break
        prior = new
    return adj, prior


class Stacker:
    """Fit on lists of (pb, pn, vis, y[, extras]) per match; predict (T, 3) probabilities.
    extras: further model probabilities (e.g. CatBoost), used by the meta-refiner only
    (vector-scaled like the others)."""

    def __init__(self, mode="meta", w=0.25, stride=5, C_reg=1.0, calibrate=True, em=False):
        self.mode, self.w, self.stride, self.C_reg = mode, w, stride, C_reg
        self.calibrate, self.em = calibrate, em

    def _rows(self, items):
        pbs, pns, vs, ys = [], [], [], []
        for pb, pn, vis, y, *_ in items:
            idx = np.flatnonzero(y != C.X)[::self.stride]
            pbs.append(pb[idx]), pns.append(pn[idx]), vs.append(vis[idx]), ys.append(y[idx].astype(int))
        return np.concatenate(pbs), np.concatenate(pns), np.concatenate(vs), np.concatenate(ys)

    def fit(self, items):
        pb, pn, vis, y = self._rows(items)
        self.th_b = vector_scale_fit(pb, y) if self.calibrate else np.zeros(3)
        self.th_n = vector_scale_fit(pn, y) if self.calibrate else np.zeros(3)
        self.prior_train = np.bincount(y, minlength=3) / len(y)
        n_extra = len(items[0][4]) if len(items[0]) > 4 else 0
        self.th_x = []
        for j in range(n_extra):
            px = np.concatenate([it[4][j][np.flatnonzero(it[3] != C.X)[::self.stride]] for it in items])
            self.th_x.append(vector_scale_fit(px, y) if self.calibrate else np.zeros(3))
        if self.mode == "meta":
            # features are built on full sequences (rolling windows need neighbours)
            Xs, ys = [], []
            for it in items:
                pb_f, pn_f, vis_f, y_f = it[:4]
                ex = [vector_scale_apply(e, th) for e, th in zip(it[4] if len(it) > 4 else (), self.th_x)]
                F = meta_features(vector_scale_apply(pb_f, self.th_b), vector_scale_apply(pn_f, self.th_n), vis_f, ex)
                idx = np.flatnonzero(y_f != C.X)[::self.stride]
                Xs.append(F[idx]), ys.append(y_f[idx].astype(int))
            X, yy = np.concatenate(Xs), np.concatenate(ys)
            self.scaler = StandardScaler().fit(X)
            self.lr = LogisticRegression(C=self.C_reg, max_iter=1000).fit(self.scaler.transform(X), yy)
        return self

    def predict(self, pb, pn, vis, extras=()):
        cb, cn = vector_scale_apply(pb, self.th_b), vector_scale_apply(pn, self.th_n)
        if self.mode == "pool":
            out = log_pool(cb, cn, self.w)
        else:
            ex = [vector_scale_apply(e, th) for e, th in zip(extras, self.th_x)]
            out = self.lr.predict_proba(self.scaler.transform(meta_features(cb, cn, vis, ex)))
        if self.em:
            out, _ = em_prior_shift(out, self.prior_train)
        return np.round(out / out.sum(1, keepdims=True), 6)


def nested_dfl(items_by_match, make):
    """{m: probs} where the stacker for m is fitted on every other match."""
    out = {}
    for m in items_by_match:
        st = make().fit([items_by_match[o] for o in items_by_match if o != m])
        it = items_by_match[m]
        out[m] = st.predict(it[0], it[1], it[2], it[4] if len(it) > 4 else ())
    return out
