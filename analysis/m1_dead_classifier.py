"""Can a classifier on tracking-only features find the dead ball better than the two-signal rule?

    uv run python analysis/m1_dead_classifier.py

Exploration, not a submission. Features are computed from a Match alone (px, py, bx, by, half,
pitch), so the same function runs on our ATD. Cross-dataset: train on degraded DFL, test on degraded
Metrica, and the other way round; the public ground truth plays no part. Offline labelling is
allowed to look forward in time, so the features do (time to the next ball sighting, centred medians).

Scored twice: frame-level balanced accuracy of dead vs alive, and the harness (spellF1) with the
predicted dead ball fed to the control rule, next to own_dead and the perfect dead ball.
"""

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.harness import evaluate  # noqa: E402
from mtah.labelers import control, own_dead  # noqa: E402

# Params
FPS = 25
CAP = 30 * FPS          # cap on gap ages, frames
MIN_BLOCK = 2 * FPS
SEED = 0


def _roll(x, w, fn="median"):
    r = pd.Series(x).rolling(w, center=True, min_periods=1)
    return getattr(r, fn)().values


def features(m):
    L, W = m["pitch"]
    n = len(m["bx"])
    X, Y = m["px"].astype(float) * L, m["py"].astype(float) * W
    bX, bY = m["bx"].astype(float) * L, m["by"].astype(float) * W
    seen = ~np.isnan(bX)
    idx = np.arange(n)
    last = pd.Series(np.where(seen, idx, np.nan)).ffill().values
    nxt = pd.Series(np.where(seen, idx, np.nan)).bfill().values
    age = np.minimum(np.nan_to_num(idx - last, nan=CAP), CAP)
    until = np.minimum(np.nan_to_num(nxt - idx, nan=CAP), CAP)
    lX = pd.Series(bX).ffill().values
    lY = pd.Series(bY).ffill().values
    bsp = np.r_[np.nan, np.hypot(np.diff(bX), np.diff(bY)) * FPS]
    lsp = pd.Series(bsp).ffill().values
    lag = 5
    vx = np.full_like(X, np.nan)
    vy = vx.copy()
    vx[lag:], vy[lag:] = (X[lag:] - X[:-lag]) * FPS / lag, (Y[lag:] - Y[:-lag]) * FPS / lag
    sp = np.hypot(vx, vy)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tx, ty = lX[:, None] - X, lY[:, None] - Y
        tn = np.hypot(tx, ty)
        cos = (vx * tx + vy * ty) / (sp * tn)
        mov = sp > 1.0
        align = np.nanmean(np.where(mov & (tn > 2), cos, np.nan), axis=1)
        moving = np.sum(mov, axis=1) / np.maximum(1, np.sum(~np.isnan(sp), axis=1))
        med = np.nanmedian(sp, axis=1)
        p90 = np.nanpercentile(sp, 90, axis=1)
        nvis = np.sum(~np.isnan(X), axis=1)
        sx, sy = np.nanstd(X, axis=1), np.nanstd(Y, axis=1)
        dball = np.nanmin(tn, axis=1)
    f = {
        "missing": (~seen).astype(float), "miss_2s": _roll((~seen).astype(float), 2 * FPS, "mean"),
        "miss_6s": _roll((~seen).astype(float), 6 * FPS, "mean"), "age": age, "until": until,
        "gap": np.where(seen, 0, age + until),
        "last_touch_d": np.minimum(lY, W - lY), "last_goal_d": np.minimum(lX, L - lX),
        "last_off": ((lX < 0) | (lX > L) | (lY < 0) | (lY > W)).astype(float),
        "last_speed": np.minimum(np.nan_to_num(lsp, nan=0), 40),
        "ball_speed_2s": _roll(np.minimum(np.nan_to_num(bsp, nan=0), 40), 2 * FPS),
        "align_2s": _roll(align, 2 * FPS), "align_6s": _roll(align, 6 * FPS),
        "moving_2s": _roll(moving, 2 * FPS), "speed_2s": _roll(med, 2 * FPS), "speed_6s": _roll(med, 6 * FPS),
        "p90_2s": _roll(p90, 2 * FPS), "nvis": nvis, "spread_x": _roll(sx, 2 * FPS), "spread_y": _roll(sy, 2 * FPS),
        "near_ball": _roll(dball, 2 * FPS),
    }
    return pd.DataFrame(f).fillna(-1)


def to_blocks(p):
    """Probability -> dead mask: 2 s median, 0.5, gaps < 2 s closed, blocks < 2 s dropped."""
    dead = _roll(p, MIN_BLOCK) > 0.5
    e = np.diff(np.r_[0, (~dead).astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if z - a < MIN_BLOCK and a > 0 and z < len(dead):
            dead[a:z] = True
    e = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if z - a < MIN_BLOCK:
            dead[a:z] = False
    return dead


def table(ms):
    Xs, ys = [], []
    for m in ms:
        g = m["half"] > 0
        Xs.append(features(m)[g])
        ys.append(m["labels"][g] == pm.D)
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def main():
    dfl_ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    src = {"DFL": [degrade(from_dfl(i), rates_for(i), seed=SEED) for i in dfl_ids],
           "Metrica": [degrade(from_metrica_sample(g), rates_for("metrica"), seed=SEED) for g in (1, 2)]}
    for train, test in (("DFL", "Metrica"), ("Metrica", "DFL")):
        Xtr, ytr = table(src[train])
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)
        clf.fit(Xtr, ytr)
        print("\ntrain {} (degraded) -> test {} (degraded)".format(train, test))
        rows = []
        for m in src[test]:
            g = m["half"] > 0
            p = np.zeros(len(g))
            p[g] = clf.predict_proba(features(m)[g])[:, 1]
            dead_clf = to_blocks(p) & g
            dead_own = own_dead(m)
            truth = m["labels"] == pm.D
            ba = lambda d: (np.mean(d[g & truth]) + np.mean(~d[g & ~truth])) / 2  # noqa: E731
            rows.append({"name": m["name"], "ba_own": ba(dead_own), "ba_clf": ba(dead_clf),
                         "own": evaluate(lambda x: control(x, dead=dead_own), [m], bootstrap=0).spell_f1[0],
                         "clf": evaluate(lambda x: control(x, dead=dead_clf), [m], bootstrap=0).spell_f1[0],
                         "perfect": evaluate(lambda x: control(x, dead="truth"), [m], bootstrap=0).spell_f1[0]})
        r = pd.DataFrame(rows)
        print("   dead vs alive balanced accuracy: own_dead {:.3f}, classifier {:.3f}".format(r.ba_own.mean(), r.ba_clf.mean()))
        print("   control rule spellF1: own_dead {:.3f} ± {:.3f}, classifier {:.3f} ± {:.3f}, perfect dead ball {:.3f}".format(
            r.own.mean(), r.own.std(ddof=0), r.clf.mean(), r.clf.std(ddof=0), r.perfect.mean()))


if __name__ == "__main__":
    main()
