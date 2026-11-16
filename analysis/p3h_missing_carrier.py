"""P3: give back the ball carrier the ATD dropped. GT-free.

    uv run python analysis/p3h_missing_carrier.py      # cache/extra_players.npz, then the comparison

On frames where the tactical ATD sees the ball, a YOLO person (conf >= 0.3) whose foot point is
within NEAR_M of the ball and farther than MATCH_M from every ATD slot is a player the ATD lost
(the overnight check: 4 of 9 such cases were the keeper, 3 the ball carrier). Its team comes from
kit colour (mtah.kits, trained on the sampled, slot-labelled boxes); referees and unsure boxes
(top class probability < P_MIN) are left out. The extra players go into two spare slots per team
of a copy of our Match, and the control rule (dead ball from the raw-ATD classifier) is compared
with and without them on the Smart Tagging side, outside the public window.
"""

import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import asset_paths  # noqa: E402
from mtah.data import GK_SLOTS, PITCH, load_tactical  # noqa: E402
from mtah.kits import fit_kit_model, kit_proba, slot_labels, torso_features  # noqa: E402

# Params
NEAR_M, MATCH_M, P_MIN = 2.0, 1.0, 0.8
OUT = os.path.join(ROOT, "cache", "extra_players.npz")


def to_pitch(H, u, v):
    q = np.einsum("nij,nj->ni", np.linalg.inv(H), np.c_[u, v, np.ones(len(u))])
    return q[:, 0] / q[:, 2] * PITCH[0], q[:, 1] / q[:, 2] * PITCH[1]


def extract():
    m = load_tactical()
    d = np.load(os.path.join(ROOT, "cache", "kit_features.npz"))
    X, Y = to_pitch(m["hom"][d["frame"]], d["x"], d["y"] + d["h"] / 2)
    lab = slot_labels(np.c_[X, Y], m["px"][d["frame"]] * PITCH[0], m["py"][d["frame"]] * PITCH[1], m["slot_team"], GK_SLOTS)
    model = fit_kit_model(d["feat"], lab)
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(ROOT, "cache", "yolo", "chunk_*.npz")))]
    c = {k: np.concatenate([p[k] for p in parts]) for k in ("frame", "cls", "conf", "x", "y", "w", "h")}
    keep = (c["cls"] == 0) & (c["conf"] >= 0.3)
    B = {k: v[keep] for k, v in c.items()}
    f = B["frame"]
    fx, fy = to_pitch(m["hom"][f], B["x"], B["y"] + B["h"] / 2)
    bX, bY = m["bx"][f] * PITCH[0], m["by"][f] * PITCH[1]
    near = np.hypot(fx - bX, fy - bY) < NEAR_M
    D = np.hypot(m["px"][f] * PITCH[0] - fx[:, None], m["py"][f] * PITCH[1] - fy[:, None])
    unmatched = np.nanmin(np.where(np.isnan(D), np.inf, D), axis=1) > MATCH_M
    cand = np.flatnonzero(near & unmatched)
    print("candidate boxes:", len(cand), "on", len(np.unique(f[cand])), "frames")
    want = {}
    for i in cand:
        want.setdefault(int(f[i]), []).append(i)
    cap = cv2.VideoCapture(asset_paths()["video"])
    rows = []
    for fr in range(max(want) + 1):
        if not cap.grab():
            break
        if fr not in want:
            continue
        img = cap.retrieve()[1]
        for i in want[fr]:
            ft = torso_features(img, B["x"][i], B["y"][i], B["w"][i], B["h"][i])
            if ft is None:
                continue
            pr = kit_proba(model, ft[None])[0]
            team = int(np.argmax([pr[0] + pr[2], pr[1] + pr[3]]))
            if max(pr[0] + pr[2], pr[1] + pr[3]) >= P_MIN:
                rows.append((fr, team, fx[i] / PITCH[0], fy[i] / PITCH[1]))
    r = np.array(rows, float).reshape(-1, 4)
    np.savez_compressed(OUT, frame=r[:, 0].astype(int), team=r[:, 1].astype(int), x=r[:, 2], y=r[:, 3])
    print("extra players kept:", len(r), "home", int((r[:, 1] == 0).sum()), "away", int((r[:, 1] == 1).sum()))


def compare():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from m1_dead_classifier import SEED, features, table, to_blocks
    from mtah.degrade import degrade, rates_for
    from mtah.external import DFL, from_dfl, from_metrica_sample
    from mtah.frames import load_frames
    from mtah.harness import evaluate
    from mtah.labelers import control
    from mtah.match import from_tactical
    from mtah.periods import NOT_PLAYED
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    train = [degrade(from_dfl(i), rates_for(i), seed=SEED) for i in ids] + \
            [degrade(from_metrica_sample(g), rates_for("metrica"), seed=SEED) for g in (1, 2)]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(*table(train))
    df = load_frames()
    v1 = from_tactical()
    g = v1["half"] > 0
    p = np.zeros(len(g))
    p[g] = clf.predict_proba(features(v1)[g])[:, 1]
    dead = (to_blocks(p) | np.isin(df["period"].values, NOT_PLAYED)) & g
    e = np.load(OUT)
    n = len(g)
    px = np.concatenate([v1["px"], np.full((n, 2), np.nan, np.float32)], axis=1)
    py = np.concatenate([v1["py"], np.full((n, 2), np.nan, np.float32)], axis=1)
    for team in (0, 1):
        s = e["team"] == team
        px[e["frame"][s], 46 + team], py[e["frame"][s], 46 + team] = e["x"][s], e["y"][s]
    plus = dict(v1, px=px, py=py, slot_team=np.r_[v1["slot_team"], [0, 1]].astype(np.int8))
    tag = np.where(df["tag_H"].values > df["tag_A"].values, pm.H, np.where(df["tag_A"].values > df["tag_H"].values, pm.A, -1))
    out = np.ones(n, bool)
    out[11185:18685] = False
    sel = g & ~dead & out & (tag >= 0)
    for name, m in (("v1", v1), ("v1 + lost carriers", plus)):
        c = np.where(g, control(m, dead=dead), pm.X).astype(np.int8)
        r = evaluate(lambda _: c, [m], bootstrap=1000)
        print("{:<20} Smart Tagging side {:.1%} | public (check) {:.3f} [{:.2f}-{:.2f}] {}/{}".format(
            name, np.mean(c[sel] == tag[sel]), r.spell_f1[0], r.boot_low[0], r.boot_high[0], r.n_pred[0], r.n_gt[0]))


if __name__ == "__main__":
    if not os.path.exists(OUT):
        extract()
    compare()
