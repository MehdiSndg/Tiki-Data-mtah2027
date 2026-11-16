"""P9bt: windows around the ball from the SoccerTrack v2 panorama videos for the possessor-crop model (external data only; the BAR-ATH
video is never used for training or tuning). Colab CPU runtime, reading the dataset from Drive; one output per half,
resumable (a half whose output exists is skipped).

Frames: every action of the 12-class ball action file (bas) at offsets -2, 0, +2 frames, and one frame every two seconds
while the ball file's status says a team has the ball (1 or 2). GSR's bbox_image is not a detection (a projection of the
pitch position, a constant ~28 x 32 px, off by 30-90 px; checked on 117092 9 Nov), so the video is cut into a WIN x WIN
window (JPEG 90) centred on the mean image position of the two players nearest the ball; persons are detected in the
windows later on the local GPU (YOLO11m, as for BAR-ATH) and matched to these projected boxes for identity and side. Per
frame: the window origin; per player among the NEAR nearest the ball: player id, side (0 left, 1 right), projected box,
distance to the ball, rank, plus the frame's ball status and the action's side and player when it is an action frame.
The ball file's y is used as it is or mirrored, whichever puts the action's player nearer the ball (median, logged).

    python p9bt_st_crops_colab.py <soccertrack root> <out dir>      # -> <out>/<match>_<half>.npz (+ _jpg.npy), <out>/log.txt
"""

import glob
import json
import os
import sys
import time

import cv2
import ijson
import numpy as np

WIN, NEAR = 640, 8
OFFSETS = (-2, 0, 2)


def log(out, *a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(os.path.join(out, "log.txt"), "a") as f:
        f.write(s + "\n")


def gsr_rows(path):
    """(frame0, pid, side, x, y, w, h, px, py) per player box; frame0 is 0-based within the half."""
    rows = []
    with open(path, "rb") as f:
        for a in ijson.items(f, "annotations.item", use_float=True):
            if a.get("supercategory") != "object":
                continue
            at = a.get("attributes") or {}
            if at.get("role") not in ("player", "goalkeeper") or at.get("team") not in ("left", "right"):
                continue
            b, p = a["bbox_image"], a["bbox_pitch"]
            pid = at.get("player_id")
            pid = int(pid) if pid not in (None, "") else -int(a.get("track_id") or 0) - 1
            rows.append((int(a["image_id"][-6:]) - 1, pid, 0 if at["team"] == "left" else 1, b["x"], b["y"], b["w"], b["h"],
                         p["x_bottom_middle"], p["y_bottom_middle"]))
    return np.array(rows, np.float64)


def half(root, mid, hname, out):
    dst = os.path.join(out, "{}_{}.npz".format(mid, hname))
    if os.path.exists(dst):
        return
    t0 = time.time()
    video = os.path.join(root, "videos", mid, "{}_panorama_{}_half.mp4".format(mid, hname))
    z = np.load(os.path.join(root, "ball", "{}_{}_ball.npz".format(mid, hname)))
    bx, by, status = z["x"].astype(np.float64), z["y"].astype(np.float64), z["status"].astype(np.int8)
    n = min(len(status), int(os.environ.get("LIMIT", 10 ** 9)))     # LIMIT: quick test on the first frames
    acts = json.load(open(os.path.join(root, "bas", mid, "{}_12_class_events.json".format(mid))))["actions"]
    hn = "1" if hname == "1st" else "2"
    acts = [a for a in acts if a["gameTime"].strip().startswith(hn + " ") and 0 <= int(a["frame"]) < n]
    R = gsr_rows(os.path.join(root, "gsr", mid, "{}_{}.json".format(mid, hname)))
    log(out, mid, hname, "gsr rows", len(R), "actions", len(acts), "parse s", round(time.time() - t0))
    fr = R[:, 0].astype(int)
    order = np.argsort(fr, kind="stable")
    R, fr = R[order], fr[order]
    start = np.searchsorted(fr, np.arange(n + 1))

    def players(k):
        return R[start[k]:start[k + 1]] if k < n else R[:0]

    # ball y orientation: the one that puts the action's player nearer the ball
    med = []
    for sgn in (1, -1):
        d = []
        for a in acts:
            k = int(a["frame"])
            P = players(k)
            m = P[:, 1] == int(a["player_id"]) if a.get("player_id") not in (None, "") else np.zeros(len(P), bool)
            if m.any() and not np.isnan(bx[k]):
                d.append(np.hypot(P[m, 7][0] - bx[k], P[m, 8][0] - sgn * by[k]))
        med.append(float(np.median(d)) if d else np.inf)
    sgn = 1 if med[0] <= med[1] else -1
    by = sgn * by
    log(out, mid, hname, "actor-ball median m, y as is / mirrored:", [round(x, 2) for x in med], "y_flipped", bool(z["y_flipped"]))

    want = {}
    for i, a in enumerate(acts):
        for o in OFFSETS:
            k = int(a["frame"]) + o
            if 0 <= k < n:
                want.setdefault(k, i)
    for k in range(0, n, 50):
        if status[k] in (1, 2):
            want.setdefault(k, -1)
    log(out, mid, hname, "frames wanted", len(want))

    buf, offs, fmeta, meta = [], [0], [], []
    cap = cv2.VideoCapture(video)
    for k in range(n):
        if k not in want:
            if not cap.grab():
                break
            continue
        ok, img = cap.read()
        if not ok:
            break
        P = players(k)
        if len(P) < 2 or np.isnan(bx[k]):
            continue
        d = np.hypot(P[:, 7] - bx[k], P[:, 8] - by[k])
        near = np.argsort(d)[:NEAR]
        H, W = img.shape[:2]
        cx = np.mean(P[near[:2], 3] + P[near[:2], 5] / 2)
        cy = np.mean(P[near[:2], 4] + P[near[:2], 6] / 2)
        x0 = int(np.clip(round(cx - WIN / 2), 0, W - WIN))
        y0 = int(np.clip(round(cy - WIN / 2), 0, H - WIN))
        ok, enc = cv2.imencode(".jpg", img[y0:y0 + WIN, x0:x0 + WIN], [cv2.IMWRITE_JPEG_QUALITY, 90])
        buf.append(enc.ravel())
        offs.append(offs[-1] + len(enc))
        ai = want[k]
        a_side = (0 if acts[ai]["team"] == "left" else 1) if ai >= 0 else -1
        a_pid = int(acts[ai]["player_id"]) if ai >= 0 and acts[ai].get("player_id") not in (None, "") else -1
        fmeta.append((k, x0, y0, status[k], a_side, a_pid, ai))
        for r, j in enumerate(near):
            meta.append((len(fmeta) - 1, P[j, 1], P[j, 2], P[j, 3], P[j, 4], P[j, 5], P[j, 6], d[j], r))
        if len(fmeta) % 500 == 0:
            log(out, mid, hname, "frame", k, "/", n, "windows", len(fmeta), "s", round(time.time() - t0))
    cap.release()
    F, M = np.array(fmeta, np.int64), np.array(meta, np.float64)
    np.save(dst[:-4] + "_jpg.tmp.npy", np.concatenate(buf))
    np.savez(dst + ".tmp.npz", offs=np.array(offs, np.int64), frame=F[:, 0], x0=F[:, 1], y0=F[:, 2], status=F[:, 3].astype(np.int8),
             a_side=F[:, 4].astype(np.int8), a_pid=F[:, 5], a_idx=F[:, 6], row=M[:, 0].astype(np.int32), pid=M[:, 1].astype(np.int64),
             side=M[:, 2].astype(np.int8), px=M[:, 3].astype(np.float32), py=M[:, 4].astype(np.float32), pw=M[:, 5].astype(np.float32),
             ph=M[:, 6].astype(np.float32), dist=M[:, 7].astype(np.float32), rank=M[:, 8].astype(np.int8), y_sign=sgn,
             a_label=np.array([a["label"] for a in acts]), a_frame=np.array([int(a["frame"]) for a in acts]))
    os.replace(dst[:-4] + "_jpg.tmp.npy", dst[:-4] + "_jpg.npy")
    os.replace(dst + ".tmp.npz", dst)
    log(out, mid, hname, "done windows", len(offs) - 1, "s", round(time.time() - t0))


def main(root, out):
    os.makedirs(out, exist_ok=True)
    for g in sorted(glob.glob(os.path.join(root, "gsr", "*", "*_*.json"))):
        mid, hname = os.path.basename(g)[:-5].split("_")
        if not os.path.exists(os.path.join(root, "videos", mid, "{}_panorama_{}_half.mp4".format(mid, hname))):
            log(out, mid, hname, "no video, skipped")
            continue
        try:
            half(root, mid, hname, out)
        except Exception as e:      # one bad half must not stop the rest
            log(out, mid, hname, "FAILED", repr(e))
    log(out, "all done")


if __name__ == "__main__":
    main(*sys.argv[1:3])
