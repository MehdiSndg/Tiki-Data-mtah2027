"""P9bx: a virtual camera over the SoccerTrack v2 panorama videos, for the video touch models (external data only; the
BAR-ATH video is never used for training or tuning). Replaces P9bt's sparse +-2-frame windows: the touch literature is
temporal (T-DEED team ball action spotting on 100-frame clips, Sanford's clip models, Peral's crop sequences), so the
frames are dense.

Camera: a WIN_W x WIN_H window at native resolution (players ~48 px tall, as in the BAR-ATH tactical video at 1920 x 1080,
where the same window can be cut around the ATD ball), centred on the mean image position of the two players nearest the
ball (GSR's projected boxes; the ball file's pitch position), held through missing ball frames and smoothed with a
centred 1 s moving average so it pans like a camera. Frames: every second frame (12.5 fps) through the half, and every
frame within +-12 frames of a ball action (25 fps around touches). JPEG 85.

Output per half in <out>/<match>_<half>/: shard_<i>.npy (concatenated JPEG bytes) + shard_<i>.npz (offsets, frames,
window origins), written every SHARD frames so a lost runtime loses at most one shard; meta.npz at the end (ball status,
ball pitch position, actions with frame / label / side / player, and every GSR player row of the stored frames: frame,
player id, side, projected box in full-image pixels, pitch position). A half with meta.npz is skipped; shards already
written are not re-encoded.

    python p9bx_st_vcam_colab.py <soccertrack root> <out dir> [<i>/<n>]      # halves i, i+n, ... (parallel processes)
"""

import glob
import json
import os
import sys
import time

import cv2
import ijson
import numpy as np

WIN_W, WIN_H, SHARD, AROUND, Q = 960, 540, 2000, 12, 85


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


def centres(R, start, n, bx, by, W, H):
    """Window origin per frame: mean projected centre of the two players nearest the ball, held, 1 s moving average."""
    c = np.full((n, 2), np.nan)
    for k in range(n):
        P = R[start[k]:start[k + 1]]
        if len(P) < 2 or np.isnan(bx[k]):
            continue
        j = np.argsort(np.hypot(P[:, 7] - bx[k], P[:, 8] - by[k]))[:2]
        c[k] = (P[j, 3] + P[j, 5] / 2).mean(), (P[j, 4] + P[j, 6] / 2).mean()
    ok = ~np.isnan(c[:, 0])
    if not ok.any():
        c[:] = W / 2, H / 2
    else:
        idx = np.maximum.accumulate(np.where(ok, np.arange(n), -1))
        first = np.argmax(ok)
        idx[idx < 0] = first
        c = c[idx]
    k = np.ones(25) / 25
    sm = np.stack([np.convolve(np.pad(c[:, i], 12, mode="edge"), k, mode="valid") for i in (0, 1)], 1)
    x0 = np.clip(np.round(sm[:, 0] - WIN_W / 2), 0, W - WIN_W).astype(np.int32)
    y0 = np.clip(np.round(sm[:, 1] - WIN_H / 2), 0, H - WIN_H).astype(np.int32)
    return x0, y0


def half(root, mid, hname, out):
    d = os.path.join(out, "{}_{}".format(mid, hname))
    if os.path.exists(os.path.join(d, "meta.npz")):
        return
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    video = os.path.join(root, "videos", mid, "{}_panorama_{}_half.mp4".format(mid, hname))
    z = np.load(os.path.join(root, "ball", "{}_{}_ball.npz".format(mid, hname)))
    bx, by, status = z["x"].astype(np.float64), z["y"].astype(np.float64), z["status"].astype(np.int8)
    n = min(len(status), int(os.environ.get("LIMIT", 10 ** 9)))
    acts = json.load(open(os.path.join(root, "bas", mid, "{}_12_class_events.json".format(mid))))["actions"]
    hn = "1" if hname == "1st" else "2"
    acts = [a for a in acts if a["gameTime"].strip().startswith(hn + " ") and 0 <= int(a["frame"]) < n]
    R = gsr_rows(os.path.join(root, "gsr", mid, "{}_{}.json".format(mid, hname)))
    R = R[np.argsort(R[:, 0], kind="stable")]
    start = np.searchsorted(R[:, 0].astype(int), np.arange(n + 1))
    # ball y as it is or mirrored: the one that puts the action's player nearer the ball (P9bt: as it is, 1.25 m)
    med = []
    for sgn in (1, -1):
        dd = []
        for a in acts:
            k = int(a["frame"])
            P = R[start[k]:start[k + 1]]
            m = P[:, 1] == int(a["player_id"]) if a.get("player_id") not in (None, "") else np.zeros(len(P), bool)
            if m.any() and not np.isnan(bx[k]):
                dd.append(np.hypot(P[m, 7][0] - bx[k], P[m, 8][0] - sgn * by[k]))
        med.append(float(np.median(dd)) if dd else np.inf)
    sgn = 1 if med[0] <= med[1] else -1
    by = sgn * by
    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, y0 = centres(R, start, n, bx, by, W, H)
    keep = np.zeros(n, bool)
    keep[::2] = True
    for a in acts:
        k = int(a["frame"])
        keep[max(k - AROUND, 0):k + AROUND + 1] = True
    frames = np.flatnonzero(keep)
    log(out, mid, hname, "rows", len(R), "actions", len(acts), "frames kept", len(frames), "of", n, "actor-ball m", [round(x, 2) for x in med],
        "prep s", round(time.time() - t0))
    shards = [frames[i:i + SHARD] for i in range(0, len(frames), SHARD)]
    done = {i for i in range(len(shards)) if os.path.exists(os.path.join(d, "shard_{}.npz".format(i)))}
    want = {int(f): i for i, s in enumerate(shards) if i not in done for f in s}
    cur = {}
    k, last = 0, max(want) if want else -1
    while k <= last and want:
        if k not in want:
            if not cap.grab():
                break
            k += 1
            continue
        ok, img = cap.read()
        if not ok:
            break
        i = want.pop(k)
        ok, enc = cv2.imencode(".jpg", img[y0[k]:y0[k] + WIN_H, x0[k]:x0[k] + WIN_W], [cv2.IMWRITE_JPEG_QUALITY, Q])
        cur.setdefault(i, []).append((k, enc.ravel()))
        if len(cur[i]) == len(shards[i]):
            fr = np.array([f for f, _ in cur[i]])
            buf = [e for _, e in cur[i]]
            offs = np.r_[0, np.cumsum([len(e) for e in buf])]
            p = os.path.join(d, "shard_{}".format(i))
            np.save(p + ".tmp.npy", np.concatenate(buf))
            os.replace(p + ".tmp.npy", p + ".npy")
            np.savez(p + ".tmp.npz", offs=offs, frame=fr, x0=x0[fr], y0=y0[fr])
            os.replace(p + ".tmp.npz", p + ".npz")
            del cur[i]
            log(out, mid, hname, "shard", i + 1, "/", len(shards), "s", round(time.time() - t0))
        k += 1
    cap.release()
    if want:
        log(out, mid, hname, "video ended early at", k, "missing", len(want))
    sel = np.isin(R[:, 0].astype(int), frames)
    np.savez(os.path.join(d, "meta.tmp.npz"), status=status[:n], bx=bx[:n], by=by[:n], y_sign=sgn, width=W, height=H,
             a_frame=np.array([int(a["frame"]) for a in acts]), a_label=np.array([a["label"] for a in acts]),
             a_side=np.array([0 if a["team"] == "left" else 1 for a in acts], np.int8),
             a_pid=np.array([int(a["player_id"]) if a.get("player_id") not in (None, "") else -1 for a in acts]),
             rows=R[sel].astype(np.float32), rows_pid=R[sel, 1].astype(np.int64))
    os.replace(os.path.join(d, "meta.tmp.npz"), os.path.join(d, "meta.npz"))
    log(out, mid, hname, "done", len(frames), "frames s", round(time.time() - t0))


def main(root, out, part="0/1"):
    i, n = (int(x) for x in part.split("/"))
    os.makedirs(out, exist_ok=True)
    halves = []
    for g in sorted(glob.glob(os.path.join(root, "gsr", "*", "*_*.json"))):
        mid, hname = os.path.basename(g)[:-5].split("_")
        if os.path.exists(os.path.join(root, "videos", mid, "{}_panorama_{}_half.mp4".format(mid, hname))):
            halves.append((mid, hname))
    for mid, hname in halves[i::n]:
        try:
            half(root, mid, hname, out)
        except Exception as e:      # one bad half must not stop the rest
            log(out, mid, hname, "FAILED", repr(e))
    log(out, "part", part, "all done")


if __name__ == "__main__":
    main(*sys.argv[1:4])
