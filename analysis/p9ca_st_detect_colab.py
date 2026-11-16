"""P9ca: person boxes with identity on the SoccerTrack v2 virtual-camera frames of P9bx (external data only). Colab GPU.

YOLO11m (COCO), persons at confidence 0.30 at the window's native scale (imgsz 960), as for BAR-ATH (analysis/
p3_yolo_pass.py: YOLO11m on the 1920 frame at imgsz 1920, persons from 0.30). GSR's projected boxes (P9bx meta rows,
off by tens of pixels) are matched to the detections as in analysis/p9bu_st_detect_crops.py: a greedy match within 120 px
gives the frame's median offset, then the shifted boxes are matched again within 60 px, one detection per player.
Output per half: <out>/<match>_<half>.npz with one row per matched player and frame (frame, player id, side, box centre
x / y and size w / h in window pixels, confidence) and the per-frame match rate.

    python p9ca_st_detect_colab.py <p9bx dir> <out dir> [<i>/<n>]
"""

import glob
import os
import sys
import time

import cv2
import numpy as np

WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolo11m.pt")
CONF, IMGSZ, FIRST, SECOND, B = 0.30, 960, 120.0, 60.0, 32


def greedy(a, b, cut):
    if not len(a) or not len(b):
        return []
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    out, ua, ub = [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(d, axis=None), d.shape)):
        if d[i, j] > cut:
            break
        if i not in ua and j not in ub:
            out.append((i, j))
            ua.add(i)
            ub.add(j)
    return out


def match(proj, det):
    p1 = greedy(proj, det, FIRST)
    if not p1:
        return {}
    off = np.median([det[j] - proj[i] for i, j in p1], axis=0)
    return dict(greedy(proj + off, det, SECOND))


def half(d, out, model):
    name = os.path.basename(d)
    dst = os.path.join(out, name + ".npz")
    if os.path.exists(dst) or not os.path.exists(os.path.join(d, "meta.npz")):
        return
    t0 = time.time()
    meta = np.load(os.path.join(d, "meta.npz"))
    R = meta["rows"].astype(np.float64)
    fr = R[:, 0].astype(int)
    order = np.argsort(fr, kind="stable")
    R, fr = R[order], fr[order]
    pid = meta["rows_pid"][order]
    rows, rate = [], []
    for p in sorted(glob.glob(os.path.join(d, "shard_*.npz")), key=lambda q: int(q.rsplit("_", 1)[1][:-4])):
        z = np.load(p)
        J = np.load(p[:-4] + ".npy", mmap_mode="r")
        offs = z["offs"]
        for s in range(0, len(z["frame"]), B):
            idx = range(s, min(s + B, len(z["frame"])))
            ims = [cv2.imdecode(np.asarray(J[offs[j]:offs[j + 1]]), cv2.IMREAD_COLOR) for j in idx]
            res = model.predict(ims, imgsz=IMGSZ, conf=CONF, classes=[0], verbose=False, half=True)
            for j, r in zip(idx, res):
                f, x0, y0 = int(z["frame"][j]), int(z["x0"][j]), int(z["y0"][j])
                a, b = np.searchsorted(fr, f), np.searchsorted(fr, f, side="right")
                P = R[a:b]
                bx = r.boxes.xywh.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                # players whose projected box centre falls inside the window (with a margin)
                cx, cy = P[:, 3] + P[:, 5] / 2 - x0, P[:, 4] + P[:, 6] / 2 - y0
                inside = (cx > -60) & (cx < 960 + 60) & (cy > -60) & (cy < 540 + 60)
                q = np.flatnonzero(inside)
                m = match(np.c_[cx[q], cy[q]], bx[:, :2])
                rate.append((f, len(q), len(m)))
                for pi, di in m.items():
                    k = a + q[pi]
                    rows.append((f, pid[k], R[k, 2], *bx[di], cf[di]))
    M = np.array(rows, np.float64)
    rt = np.array(rate)
    np.savez_compressed(dst, frame=M[:, 0].astype(np.int32), pid=M[:, 1].astype(np.int64), side=M[:, 2].astype(np.int8),
                        box=M[:, 3:7].astype(np.float32), conf=M[:, 7].astype(np.float32), rate=rt)
    print(name, "frames", len(rt), "matched {:.1%}".format(rt[:, 2].sum() / max(rt[:, 1].sum(), 1)), "s", round(time.time() - t0), flush=True)


def main(src, out, part="0/1"):
    from ultralytics import YOLO
    i, n = (int(x) for x in part.split("/"))
    model = YOLO(WEIGHTS)
    os.makedirs(out, exist_ok=True)
    for d in sorted(glob.glob(os.path.join(src, "*_*")))[i::n]:
        if os.path.isdir(d):
            half(d, out, model)


if __name__ == "__main__":
    main(*sys.argv[1:4])
