"""Kit colour: which team a YOLO person box belongs to, and where the keeper the ATD lost is. (P3)

    from mtah.kits import torso_features, slot_labels, fit_kit_model, kit_proba, recover_keeper

Labels come from the ATD, not from a person: a box whose foot point is within MATCH_M of exactly one
slot (the next slot farther than CLEAR_M) takes that slot's class -- 0 home, 1 away, 2 home keeper,
3 away keeper. On our match the classifier tells the teams apart on 97% of single boxes across
halves; the two keepers' kits are alike, so the end of the pitch decides which keeper it is
(analysis/p3c_kit_classifier.py, notes/BULGULAR.md).
"""

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# Params
MATCH_M, CLEAR_M = 1.0, 1.5
P_KEEPER = 0.5
TOUCH_M = 2.0           # the touchline strip holds assistant referees, ball boys and staff


def torso_features(img, x, y, w, h):
    """Colour of a box's torso -- central half in width, 15-50% of the height -- without grass.

    (19,): median L*a*b* (3), hue histogram of saturated pixels (12), white / dark / saturated
    share and the pixel count (4). None when nothing but grass is left.
    """
    x0, x1 = int(x - w / 4), int(x + w / 4)
    y0, y1 = int(y - h / 2 + 0.15 * h), int(y - h / 2 + 0.50 * h)
    x0, y0 = max(0, x0), max(0, y0)
    crop = img[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    keep = ~((hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] > 50))
    if keep.sum() < 4:
        return None
    hv, lv = hsv[keep], lab[keep]
    sat = hv[:, 1] > 60
    hist = np.histogram(hv[sat, 0], bins=12, range=(0, 180))[0].astype(float)
    hist /= max(1.0, hist.sum())
    sv = [np.mean((hv[:, 1] < 40) & (hv[:, 2] > 170)), np.mean(hv[:, 2] < 60), np.mean(sat), float(keep.sum())]
    return np.r_[np.median(lv, axis=0), hist, sv]


def slot_labels(boxes_m, px_m, py_m, slot_team, gk_slots):
    """Class per box from the ATD: boxes_m (n, 2) foot points in metres; px_m, py_m (n, k) the slots
    on each box's frame, metres. -1 when no slot is clearly the box's."""
    D = np.hypot(px_m - boxes_m[:, :1], py_m - boxes_m[:, 1:2])
    D = np.where(np.isnan(D), np.inf, D)
    k = np.argmin(D, axis=1)
    near = D[np.arange(len(k)), k]
    second = np.sort(D, axis=1)[:, 1] if D.shape[1] > 1 else np.full(len(k), np.inf)
    lab = np.full(len(k), -1)
    ok = (near < MATCH_M) & (second > CLEAR_M)
    lab[ok] = slot_team[k[ok]]
    lab[ok & (k == gk_slots[0])] = 2
    lab[ok & (k == gk_slots[1])] = 3
    return lab


def fit_kit_model(features, labels, seed=0):
    keep = labels >= 0
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=seed)
    return clf.fit(features[keep], labels[keep])


def kit_proba(model, features):
    """(n, 4) probabilities: home, away, home keeper, away keeper."""
    return model.predict_proba(features)


def recover_keeper(X, Y, p_keeper, own_half_low_x, pitch):
    """Index of the box most likely to be the keeper -- keeper kit, on the pitch, off the touchline
    strip, in the keeper's own half -- or -1."""
    L, W = pitch
    ok = (p_keeper > P_KEEPER) & (X > 0) & (X < L) & (Y > TOUCH_M) & (Y < W - TOUCH_M)
    ok &= (X < L / 2) if own_half_low_x else (X > L / 2)
    if not ok.any():
        return -1
    return int(np.flatnonzero(ok)[np.argmax(p_keeper[ok])])
