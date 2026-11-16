"""EDA section B: is the ATD ball where the ball really is? A systematic audit against the video.

    uv run python analysis/b_atd_ball_audit.py [--n 1000]

For a sample of graded frames on which the ATD reports a ball, spread evenly over the match:

  1. crop the video around the projected ATD ball at native resolution and run YOLO on the crop,
     upscaled, where a few-pixel ball is far easier to see than in the full frame;
  2. run YOLO on the full frame to find a ball anywhere else on the pitch;
  3. classify each frame:
       confirmed     a ball within tolerance of the ATD point (a lofted ball shows ABOVE its
                     ground point in the image, so the window extends upwards)
       contradicted  no ball at the ATD point, but a confident ball elsewhere on the pitch
       undetermined  no ball found anywhere (occluded, blurred, in the air, or YOLO missed it)

YOLO is not ground truth either, so the contradicted frames are written out for a visual check:
eda_out/B/audit/. Results table: eda_out/B/atd_ball_audit.csv.
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402
from mtah.data import FPS, PITCH, load_tactical, project  # noqa: E402

# Params
OUT = os.path.join(ROOT, "eda_out", "B")
CROP = 320                 # half-size of the native-resolution crop around the ATD ball, px
UPSCALE = 2                # crop is upscaled before detection
TOL_X, TOL_DOWN, TOL_UP = 30, 30, 160    # px: |dx|, below, above the ATD point (lofted balls)
CONF_LOCAL, CONF_GLOBAL = 0.25, 0.45
SEED = 0


def detect(model, img, conf, imgsz):
    r = model.predict(img, imgsz=imgsz, conf=conf, classes=[32], verbose=False)[0]
    return r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    from ultralytics import YOLO

    os.makedirs(os.path.join(OUT, "audit"), exist_ok=True)
    m = load_tactical()
    seen = ~np.isnan(m["bx"])
    pool = np.flatnonzero(m["graded"] & seen)
    # evenly spread over the match: one random frame from each of n equal slices of the pool
    rng = np.random.RandomState(SEED)
    edges = np.linspace(0, len(pool), args.n + 1).astype(int)
    frames = np.array([pool[rng.randint(a, b)] for a, b in zip(edges[:-1], edges[1:])])

    step = np.full(len(m["bx"]), np.nan)
    step[1:] = np.hypot(np.diff(m["bx"]) * PITCH[0], np.diff(m["by"]) * PITCH[1]) * FPS
    model = YOLO(os.path.join(ROOT, "models", "yolo11m.pt"))
    cap = cv2.VideoCapture(asset_paths()["video"])
    rows, shown = [], 0
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        hom = m["hom"][f]
        ax, ay = project(hom, m["bx"][f], m["by"][f])
        inside_img = 0 <= ax < img.shape[1] and 0 <= ay < img.shape[0]
        on_pitch = 0 <= m["bx"][f] <= 1 and 0 <= m["by"][f] <= 1

        # 1. local search around the ATD point
        local = None
        if inside_img:
            x0 = int(np.clip(ax - CROP, 0, img.shape[1] - 2 * CROP))
            y0 = int(np.clip(ay - CROP, 0, img.shape[0] - 2 * CROP))
            crop = cv2.resize(img[y0:y0 + 2 * CROP, x0:x0 + 2 * CROP], None, fx=UPSCALE, fy=UPSCALE,
                              interpolation=cv2.INTER_CUBIC)
            xywh, conf = detect(model, crop, CONF_LOCAL, 2 * CROP * UPSCALE)
            for (x, y, w, h), c in zip(xywh, conf):
                dx, dy = x0 + x / UPSCALE - ax, y0 + y / UPSCALE - ay
                if abs(dx) <= TOL_X and -TOL_UP <= dy <= TOL_DOWN and (local is None or c > local[2]):
                    local = (dx, dy, float(c))

        # 2. global search for a confident ball on the pitch, away from the ATD point
        other = None
        if local is None:
            inv = np.linalg.inv(hom)
            xywh, conf = detect(model, img, CONF_GLOBAL, 1920)
            for (x, y, w, h), c in zip(xywh, conf):
                u, v = project(inv, x, y + h / 2)
                if -0.01 <= u <= 1.01 and -0.01 <= v <= 1.01 and (other is None or c > other[3]):
                    other = (x, y, np.hypot((u - m["bx"][f]) * PITCH[0], (v - m["by"][f]) * PITCH[1]), float(c))

        status = "confirmed" if local else ("contradicted" if other else "undetermined")
        rows.append((int(f), status, inside_img, on_pitch, step[f],
                     local[1] if local else np.nan, other[2] if other else np.nan))
        if status == "contradicted" and shown < 24:
            vis = img.copy()
            cv2.circle(vis, (int(ax), int(ay)), 16, (255, 255, 255), 2)
            cv2.putText(vis, "ATD", (int(ax) + 18, int(ay)), 0, 0.8, (255, 255, 255), 2)
            cv2.rectangle(vis, (int(other[0]) - 14, int(other[1]) - 14), (int(other[0]) + 14, int(other[1]) + 14),
                          (0, 255, 255), 2)
            cv2.putText(vis, "YOLO %.2f  %.0f m away" % (other[3], other[2]), (int(other[0]) + 16, int(other[1]) - 8),
                        0, 0.8, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(OUT, "audit", "contra_{:06d}.jpg".format(f)),
                        cv2.resize(vis, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])
            shown += 1

    with open(os.path.join(OUT, "atd_ball_audit.csv"), "w") as handle:
        handle.write("frame,status,inside_image,on_pitch,atd_speed,local_dy,other_dist_m\n")
        for r in rows:
            handle.write("{},{},{},{},{:.2f},{:.1f},{:.1f}\n".format(r[0], r[1], int(r[2]), int(r[3]), r[4], r[5], r[6]))

    status = np.array([r[1] for r in rows])
    speed = np.array([r[4] for r in rows])
    onp = np.array([r[3] for r in rows])
    print("ATD-ball frames audited: {} (spread over the graded match)".format(len(rows)))
    for s in ("confirmed", "contradicted", "undetermined"):
        print("   {:<13} {:5d}  {:5.1%}".format(s, int((status == s).sum()), (status == s).mean()))
    print("\nby ATD ball speed:")
    for lo, hi in ((0, 2), (2, 10), (10, 25), (25, 1e9)):
        k = (speed >= lo) & (speed < hi)
        if k.sum():
            print("   {:>4}-{:<4} m/s  n={:4d}  confirmed {:5.1%}  contradicted {:5.1%}  undetermined {:5.1%}".format(
                lo, "inf" if hi > 1e8 else hi, int(k.sum()), (status[k] == "confirmed").mean(),
                (status[k] == "contradicted").mean(), (status[k] == "undetermined").mean()))
    for name, k in (("ATD ball on the pitch", onp), ("ATD ball OFF the pitch", ~onp)):
        if k.sum():
            print("   {:<23} n={:4d}  confirmed {:5.1%}  contradicted {:5.1%}".format(
                name, int(k.sum()), (status[k] == "confirmed").mean(), (status[k] == "contradicted").mean()))
    dy = np.array([r[5] for r in rows])
    dy = dy[~np.isnan(dy)]
    print("\nconfirmed: vertical offset of the video ball from the ATD point, px: p10 {:.0f}  p50 {:.0f}  p90 {:.0f}"
          .format(*np.percentile(dy, [10, 50, 90])))
    d = np.array([r[6] for r in rows])
    d = d[~np.isnan(d)]
    if len(d):
        print("contradicted: distance on the pitch between ATD ball and the YOLO ball, m: p25 {:.0f}  p50 {:.0f}  p75 {:.0f}"
              .format(*np.percentile(d, [25, 50, 75])))


if __name__ == "__main__":
    main()
