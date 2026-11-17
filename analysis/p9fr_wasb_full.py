"""P9fr: WASB (pretrained on soccer, ISSIA-CNR; no training on any frame of ours) over every graded frame of the tactical
video, as its authors run it: 3 consecutive frames in, the middle frame's heatmap, blob score threshold 0.5, then
WASB's own online tracker (trackers/online.py, max_disp 300 px), refreshed at each half's start. Our ball is ~8 px in
the 1920 px frame, so frames are cut into 2 x 2 overlapping tiles resized to WASB's 512 x 288 input (ball ~4 px, the
scale WASB was trained at; P5l's measurement: whole frame 0%). Inference only; strict fp32 not required.

    WASB_SRC=<WASB-SBDT src> WASB_W=<dir with wasb_soccer_best.pth.tar> python analysis/p9fr_wasb_full.py [N_FRAMES]
    # -> cache/wasb_full.npz: per video frame x, y (px), visible, score, and X, Y on the pitch (m) via the homography

Use (P9fr, label-free, registered 17 Nov): fill the ATD ball only on graded frames where the ATD has none and the
tracked WASB ball is visible, on the pitch and within 30 m of the last ATD ball; judged by agreement with the video's
team calls where the video is sure, and with Smart Tagging's set pieces (restart recall), both not lower.
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
from p5l_wasb import Runner, load  # noqa: E402

THR, MAX_DISP, BATCH = 0.5, 300.0, int(os.environ.get("WASB_BATCH", 16))


def main(limit=None):
    import torch
    from games import asset_paths
    from mtah.data import PITCH, load_tactical
    src, wdir = os.environ["WASB_SRC"], os.environ["WASB_W"]
    sys.path.insert(0, src)
    if "Inf" not in np.__dict__:
        np.Inf = np.inf                                         # WASB's tracker predates NumPy 2
    from trackers.online import OnlineTracker
    model, cfg = load(src, wdir, "wasb")
    run = Runner(model, cfg, 2.0)
    m = load_tactical()
    g, hom = m["graded"], m["hom"]
    n = len(g)
    X = np.full((n, 6), np.nan, np.float32)                 # x, y, visible, score, X, Y
    cap = cv2.VideoCapture(asset_paths()["video"])
    t0 = time.time()
    edges = np.diff(np.r_[0, g.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if limit:
            b = min(b, a + int(limit))
        tracker = OnlineTracker({"tracker": {"max_disp": MAX_DISP}})
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(a - 1, 0))
        preps = {}
        f = max(a - 1, 0)
        ok, img = cap.read()
        preps[f] = run.prep(img)
        targets = []
        for f in range(max(a - 1, 0) + 1, b + 1):
            ok, img = cap.read()
            if not ok:
                break
            preps[f] = run.prep(img)
            t = f - 1
            if t >= a and t - 1 in preps:
                targets.append(t)
            if len(targets) == BATCH or (f == b and targets):
                x = np.stack([np.concatenate([preps[t - 1], preps[t], preps[t + 1]], axis=1) for t in targets])  # (B, tiles, 9, h, w)
                B, T = x.shape[:2]
                with torch.no_grad():
                    hm = torch.sigmoid(run.model(torch.from_numpy(x.reshape(B * T, *x.shape[2:])).cuda())[0][:, run.target]).cpu().numpy()
                hm = hm.reshape(B, T, *hm.shape[1:])
                for t, h in zip(targets, hm):
                    from p5l_wasb import blobs, merge
                    cands = []
                    for (tx, ty, tw, th), hh in zip(run.tiles, h):
                        for c, bx, by in blobs(hh):
                            if c >= THR:
                                cands.append((c, tx + (bx + 0.5) * tw / run.ow - 0.5, ty + (by + 0.5) * th / run.oh - 0.5))
                    dets = [{"xy": np.array([cx, cy]), "score": c} for c, cx, cy in merge(cands)]
                    r = tracker.update(dets)
                    if r["visi"]:
                        p = np.linalg.inv(hom[t]) @ np.array([r["x"], r["y"], 1.0])
                        X[t] = (r["x"], r["y"], 1, r["score"], p[0] / p[2] * PITCH[0], p[1] / p[2] * PITCH[1])
                    else:
                        X[t, 2] = 0
                for t in targets:
                    preps.pop(t - 1, None)
                targets = []
        print("half done", a, b, "{:.0f} s".format(time.time() - t0), "visible share {:.3f}".format(np.nanmean(X[a:b, 2])), flush=True)
    out = os.path.join(ROOT, "cache", "wasb_full.npz" if not limit else "wasb_smoke.npz")
    np.savez_compressed(out, x=X[:, 0], y=X[:, 1], visible=X[:, 2], score=X[:, 3], X=X[:, 4], Y=X[:, 5])
    print("->", out, "{:.1f} frames/s".format(g.sum() / (time.time() - t0) if not limit else 2 * int(limit) / (time.time() - t0)))


if __name__ == "__main__":
    main(*sys.argv[1:2])
