"""P9cl: the virtual-camera frames of our match's tactical video for the video possession model (inference input only;
nothing is trained, fitted or tuned here, and no frame is looked at). The windows of P9cc (cache/barath_vcam_plan.npz:
960 x 540 around the two players nearest the ATD ball), every second frame of each half (12.5 fps, as the SoccerTrack
frames of P9bx the model is trained on), JPEG 85, in P9bx's shard layout so the same reader runs on both:

    cache/barath_vcam/BARATH_<1st|2nd>/shard_<i>.npy + shard_<i>.npz (offs, frame within the half, x0, y0)
    cache/barath_vcam/BARATH_<1st|2nd>/meta.npz (status: zeros, one per frame of the half; no actions; vid0: the video
    frame of the half's frame 0)

    python analysis/p9cl_barath_vcam_extract.py
"""

import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

WIN_W, WIN_H, SHARD, Q = 960, 540, 2000, 85
OUT = os.path.join(ROOT, "cache", "barath_vcam")


def flush(d, i, buf, fr, x0, y0):
    offs = np.r_[0, np.cumsum([len(b) for b in buf])].astype(np.int64)
    p = os.path.join(d, "shard_{}".format(i))
    np.save(p + ".npy", np.concatenate(buf))
    np.savez(p + ".npz", offs=offs, frame=np.array(fr, np.int32), x0=np.array(x0, np.int16), y0=np.array(y0, np.int16))


def main():
    plan = np.load(os.path.join(ROOT, "cache", "barath_vcam_plan.npz"))
    half, X0, Y0 = plan["half"], plan["x0"], plan["y0"]
    video = os.environ.get("BARATH_VIDEO")                     # Colab: the tactical video's path; locally from mtah.data
    if not video:
        from mtah.data import asset_paths
        video = asset_paths()["video"]
    cap = cv2.VideoCapture(video)
    t0 = time.time()
    for h, hname in ((1, "1st"), (2, "2nd")):
        idx = np.flatnonzero(half == h)
        a, b = int(idx[0]), int(idx[-1]) + 1
        assert np.all(half[a:b] == h), "the half is not one contiguous block"
        d = os.path.join(OUT, "BARATH_{}".format(hname))
        if os.path.exists(os.path.join(d, "meta.npz")):
            continue
        os.makedirs(d, exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        buf, fr, x0, y0, si = [], [], [], [], 0
        for k in range(a, b):
            ok, img = cap.read()
            if not ok:
                raise RuntimeError("video ended at frame {}".format(k))
            f = k - a
            if f % 2:
                continue
            enc = cv2.imencode(".jpg", img[Y0[k]:Y0[k] + WIN_H, X0[k]:X0[k] + WIN_W], [cv2.IMWRITE_JPEG_QUALITY, Q])[1]
            buf.append(enc.ravel())
            fr.append(f)
            x0.append(X0[k])
            y0.append(Y0[k])
            if len(buf) == SHARD:
                flush(d, si, buf, fr, x0, y0)
                si += 1
                buf, fr, x0, y0 = [], [], [], []
                print(hname, "shard", si, "frame", f, "/", b - a, "s", round(time.time() - t0), flush=True)
        if buf:
            flush(d, si, buf, fr, x0, y0)
        np.savez(os.path.join(d, "meta.npz"), status=np.zeros(b - a, np.uint8), a_frame=np.zeros(0, np.int32),
                 a_label=np.zeros(0, "U1"), a_side=np.zeros(0, np.int8), vid0=a)
        print(hname, "done frames", b - a, "s", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
