"""P3 groundwork: kit colour tells the team and finds the keeper the ATD has lost. GT-free.

    uv run python analysis/p3c_kit_classifier.py      # needs cache/kit_features.npz (p3b)

1. labels without a person: each YOLO person box (sampled frames) is matched to an ATD slot when its
   foot point is within 1 m of exactly one slot (the next slot > 1.5 m away); the slot gives
   Barcelona / Athletic / Barcelona keeper / Athletic keeper
2. colour classifier (HistGradientBoosting on the torso features), trained on one half, tested on
   the other (lighting changes between halves)
3. keeper recovery: on frames where a keeper slot is a GHOST (mtah.keepers) during play, the person
   most likely to wear a keeper kit, on the pitch away from the touchlines, in that keeper's own
   half (the two keepers' kits are similar; the end of the pitch tells them apart)
"""

import os
import sys

import numpy as np
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402
from mtah.data import GK_SLOTS, PITCH, load_tactical  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.keepers import GHOST  # noqa: E402
from mtah.kits import fit_kit_model, recover_keeper, slot_labels  # noqa: E402
from mtah.periods import FIRST, SECOND, attack_sign  # noqa: E402

# Params
NAMES = ("Barcelona", "Athletic", "Barcelona GK", "Athletic GK")


def labels(d, m):
    f = d["frame"]
    Hi = np.linalg.inv(m["hom"][f])
    q = np.einsum("nij,nj->ni", Hi, np.c_[d["x"], d["y"] + d["h"] / 2, np.ones(len(f))])
    X, Y = q[:, 0] / q[:, 2] * PITCH[0], q[:, 1] / q[:, 2] * PITCH[1]
    lab = slot_labels(np.c_[X, Y], m["px"][f] * PITCH[0], m["py"][f] * PITCH[1], m["slot_team"], GK_SLOTS)
    return lab, X, Y


def main():
    d = np.load(os.path.join(ROOT, "cache", "kit_features.npz"))
    m = load_tactical()
    df = load_frames()
    lab, X, Y = labels(d, m)
    f, F = d["frame"], d["feat"]
    half = m["half"][f]
    print("person boxes {}, matched to one slot {} ({})".format(len(f), int((lab >= 0).sum()),
          ", ".join("{} {}".format(n, int((lab == i).sum())) for i, n in enumerate(NAMES))))
    p_gk = np.zeros(len(f))
    for tr, te in ((1, 2), (2, 1)):
        a, b = (lab >= 0) & (half == tr), half == te
        clf = fit_kit_model(F[a], lab[a])
        pr = clf.predict_proba(F[b])
        p_gk[b] = pr[:, 2] + pr[:, 3]
        t = b & (lab >= 0)
        cm = confusion_matrix(lab[t], pr[lab[b] >= 0].argmax(1), labels=[0, 1, 2, 3])
        print("train half {} -> test half {}: accuracy {:.3f}".format(tr, te, np.trace(cm) / cm.sum()))
        for i, n in enumerate(NAMES):
            print("   {:<13} recall {:.3f}  precision {:.3f}".format(n, cm[i, i] / max(1, cm[i].sum()), cm[i, i] / max(1, cm[:, i].sum())))
        gk_true = np.isin(lab[t], (2, 3))
        gk_pred = np.isin(pr[lab[b] >= 0].argmax(1), (2, 3))
        print("   any keeper   recall {:.3f}  precision {:.3f}".format(np.mean(gk_pred[gk_true]), np.mean(gk_true[gk_pred])))
    sign = attack_sign(asset_paths()["atd_meta"])
    play = np.isin(df["period"].values, [FIRST, SECOND])
    for team, side in ((0, "H"), (1, "A")):
        st = df[side + "_gk_status"].values
        frames = np.unique(f[(st[f] == GHOST) & play[f]])
        found = 0
        for fr in frames:
            s = sign[m["half"][fr]] * (1 if team == 0 else -1)
            sel = f == fr
            found += recover_keeper(X[sel], Y[sel], p_gk[sel], own_half_low_x=s > 0, pitch=PITCH) >= 0
        print("{} keeper: ghost frames in play (sampled 1 in 25) {}, a keeper-kit person found in its half {:.0%}".format(
            side, len(frames), found / max(1, len(frames))))


if __name__ == "__main__":
    main()
