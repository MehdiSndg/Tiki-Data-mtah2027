"""P9bu: person boxes and possessor crops for the SoccerTrack windows of P9bt (external data only). Local GPU.

YOLO11m (COCO), persons at confidence 0.30, at the window's native scale (imgsz 640) as for BAR-ATH (analysis/p3_yolo_pass.py
runs YOLO11m on the 1920 frame at imgsz 1920, persons from 0.30). Each projected GSR box (P9bt; off by tens of pixels) is
matched to a detection: a first greedy match within 120 px gives the window's median offset, then the boxes shifted by it
are matched again within 60 px (40 px kept 62% of the actors on 117092 1st, 60 px 72%, 80 px 78% with more risk of taking the neighbour), one detection per player. Crop as analysis/p6d_possessor_crops.py: a square around the
detection, side 1.4 x its larger side, 128 x 128, JPEG 95. Per crop: half, frame, rank (nearest the ball first), player,
side, distance to the ball, ball status, the action's side and player (action frames), detection box and confidence.

    python analysis/p9bu_st_detect_crops.py <windows dir> <out dir>     # -> <out>/<match>_<half>.npz, one per half
"""

import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEIGHTS = os.path.join(ROOT, "models", "yolo11m.pt")
SIZE, MARGIN, CONF = 128, 1.4, 0.30
FIRST, SECOND = 120.0, 60.0


def greedy(a, b, cut):
    """pairs (i, j) of rows of a and b (centres, (n, 2)) nearest first, each used once, distance <= cut."""
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
    """proj: (n, 2) projected centres; det: (m, 2) detection centres -> {proj row: det row}."""
    p1 = greedy(proj, det, FIRST)
    if not p1:
        return {}
    off = np.median([det[j] - proj[i] for i, j in p1], axis=0)
    return dict(greedy(proj + off, det, SECOND))


def crop(img, x, y, w, h):
    """x, y: box centre."""
    s = MARGIN * max(w, h)
    x0, y0 = int(round(x - s / 2)), int(round(y - s / 2))
    x1, y1 = x0 + int(round(s)), y0 + int(round(s))
    H, W = img.shape[:2]
    pad = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if sx1 > sx0 and sy1 > sy0:
        pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(pad, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def one(src, dst, model):
    z = np.load(src)
    J = np.load(src[:-4] + "_jpg.npy", mmap_mode="r")
    offs = z["offs"]
    rows_of = {}
    for r, f in enumerate(z["row"]):
        rows_of.setdefault(int(f), []).append(r)
    buf, eo, meta, nmatch, nproj = [], [0], [], 0, 0
    B = 32
    for s in range(0, len(offs) - 1, B):
        idx = list(range(s, min(s + B, len(offs) - 1)))
        ims = [cv2.imdecode(np.asarray(J[offs[i]:offs[i + 1]]), cv2.IMREAD_COLOR) for i in idx]
        res = model.predict(ims, imgsz=640, conf=CONF, classes=[0], verbose=False, half=True)
        for i, im, r in zip(idx, ims, res):
            bx = r.boxes.xywh.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            rr = rows_of.get(i, [])
            if not rr:
                continue
            proj = np.c_[z["px"][rr] + z["pw"][rr] / 2 - z["x0"][i], z["py"][rr] + z["ph"][rr] / 2 - z["y0"][i]]
            m = match(proj, bx[:, :2])
            nproj += len(rr)
            nmatch += len(m)
            for pi, di in m.items():
                ok, enc = cv2.imencode(".jpg", crop(im, *bx[di]), [cv2.IMWRITE_JPEG_QUALITY, 95])
                buf.append(enc.ravel())
                eo.append(eo[-1] + len(enc))
                q = rr[pi]
                meta.append((z["frame"][i], z["rank"][q], z["pid"][q], z["side"][q], z["dist"][q], z["status"][i], z["a_side"][i],
                             z["a_pid"][i], z["a_idx"][i], *bx[di], cf[di]))
    M = np.array(meta, np.float64)
    np.savez(dst, jpg=np.concatenate(buf), offs=np.array(eo, np.int64), frame=M[:, 0].astype(np.int32), rank=M[:, 1].astype(np.int8),
             pid=M[:, 2].astype(np.int64), side=M[:, 3].astype(np.int8), dist=M[:, 4].astype(np.float32), status=M[:, 5].astype(np.int8),
             a_side=M[:, 6].astype(np.int8), a_pid=M[:, 7].astype(np.int64), a_idx=M[:, 8].astype(np.int32), box=M[:, 9:13].astype(np.float32),
             conf=M[:, 13].astype(np.float32), a_label=z["a_label"], a_frame=z["a_frame"])
    print(os.path.basename(src), "windows", len(offs) - 1, "players", nproj, "matched {:.1%}".format(nmatch / max(nproj, 1)), flush=True)


def main(src_dir, out):
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    os.makedirs(out, exist_ok=True)
    for src in sorted(glob.glob(os.path.join(src_dir, "*_*[dt].npz"))):
        dst = os.path.join(out, os.path.basename(src))
        if not os.path.exists(dst):
            one(src, dst, model)


if __name__ == "__main__":
    main(*sys.argv[1:3])
