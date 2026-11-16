"""P3 Task 5a: where is the keeper on every frame his ATD slot is a ghost? GT-free.

    uv run python analysis/p3g_keeper_recovery.py      # cache/keeper_recovered.npz (gitignored)

The kit model (mtah.kits) is trained on all slot-labelled boxes of the sampled frames
(cache/kit_features.npz, analysis/p3b_kit_features.py). Then every frame on which a keeper slot is
GHOST (mtah.keepers) is decoded, the torso of each YOLO person in that keeper's own half is read,
and mtah.kits.recover_keeper picks the keeper. Output per (frame, team): X, Y normalised, the keeper
probability; frames where nobody qualifies are absent.
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import asset_paths  # noqa: E402
from mtah.data import GK_SLOTS, PITCH, load_tactical  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.keepers import GHOST  # noqa: E402
from mtah.kits import fit_kit_model, kit_proba, recover_keeper, slot_labels, torso_features  # noqa: E402
from mtah.periods import attack_sign  # noqa: E402

OUT = os.path.join(ROOT, "cache", "keeper_recovered.npz")


def to_pitch(hom, u, v):
    q = np.einsum("nij,nj->ni", np.linalg.inv(hom), np.c_[u, v, np.ones(len(u))])
    return q[:, 0] / q[:, 2] * PITCH[0], q[:, 1] / q[:, 2] * PITCH[1]


def main():
    m, df = load_tactical(), load_frames()
    d = np.load(os.path.join(ROOT, "cache", "kit_features.npz"))
    X, Y = to_pitch(m["hom"][d["frame"]], d["x"], d["y"] + d["h"] / 2)
    lab = slot_labels(np.c_[X, Y], m["px"][d["frame"]] * PITCH[0], m["py"][d["frame"]] * PITCH[1], m["slot_team"], GK_SLOTS)
    model = fit_kit_model(d["feat"], lab)
    sign = attack_sign(asset_paths()["atd_meta"])
    # full boxes (with width) are needed for the torso: re-read them from the YOLO cache
    import glob
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(ROOT, "cache", "yolo", "chunk_*.npz")))]
    cols = {c: np.concatenate([p[c] for p in parts]) for c in ("frame", "cls", "conf", "x", "y", "w", "h")}
    keep = (cols["cls"] == 0) & (cols["conf"] >= 0.3)
    B = {c: v[keep] for c, v in cols.items()}
    order = np.argsort(B["frame"], kind="stable")
    bf = B["frame"][order]
    ghost = {t: df[s + "_gk_status"].values == GHOST for t, s in ((0, "H"), (1, "A"))}
    todo = np.flatnonzero(ghost[0] | ghost[1])
    rows = []
    cap = cv2.VideoCapture(asset_paths()["video"])
    want = set(todo.tolist())
    for f in range(int(todo.max()) + 1):
        if not cap.grab():
            break
        if f not in want:
            continue
        img = cap.retrieve()[1]
        idx = order[np.searchsorted(bf, f):np.searchsorted(bf, f, side="right")]
        if not len(idx):
            continue
        x, y, w, h = (B[c][idx] for c in ("x", "y", "w", "h"))
        fx, fy = to_pitch(np.repeat(m["hom"][f][None], len(idx), 0), x, y + h / 2)
        for team in (0, 1):
            if not ghost[team][f]:
                continue
            low = sign[m["half"][f]] * (1 if team == 0 else -1) > 0
            side = (fx < PITCH[0] / 2) if low else (fx > PITCH[0] / 2)
            feats, j = [], []
            for k in np.flatnonzero(side):
                ft = torso_features(img, x[k], y[k], w[k], h[k])
                if ft is not None:
                    feats.append(ft)
                    j.append(k)
            if not feats:
                continue
            pr = kit_proba(model, np.array(feats))
            p_gk = pr[:, 2] + pr[:, 3]
            i = recover_keeper(fx[j], fy[j], p_gk, own_half_low_x=low, pitch=PITCH)
            if i >= 0:
                rows.append((f, team, fx[j][i] / PITCH[0], fy[j][i] / PITCH[1], p_gk[i]))
    r = np.array(rows, float).reshape(-1, 5)
    np.savez_compressed(OUT, frame=r[:, 0].astype(int), team=r[:, 1].astype(int), x=r[:, 2], y=r[:, 3], p=r[:, 4])
    for t in (0, 1):
        print("team {}: ghost frames {}, recovered {} ({:.0%})".format(t, int(ghost[t].sum()), int((r[:, 1] == t).sum()),
                                                                     (r[:, 1] == t).sum() / max(1, ghost[t].sum())))


if __name__ == "__main__":
    main()
