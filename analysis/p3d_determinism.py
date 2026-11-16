"""P3 Task 1: does the detector give the same boxes every time? (RULES.md: the submission must be
regenerated exactly.)

    uv run python analysis/p3d_determinism.py        # -> eda_out/reports/P3_determinism.md

The same 200 frames (100 from each half, evenly spaced) through YOLO11m at imgsz 1920 in four modes:
GPU fp16 twice (run-to-run), GPU fp32, CPU fp32. Compared pairwise (fp16 runs, fp16 against fp32,
GPU fp32 against CPU fp32): box count,
the largest centre and confidence change of matched boxes (same class, IoU > 0.5), and whether the
set of ball boxes at conf >= 0.3 is the same.

Historical record (22 Oct): its fp32 modes ran with torch's default cuDNN TF32 on. The strict fp32
comparison across devices is analysis/p5_gpu_bench.py (notes/BULGULAR.md).
"""

import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, asset_paths, half_video_blocks  # noqa: E402

N_PER_HALF = 100
OUT = os.path.join(ROOT, "eda_out", "reports", "P3_determinism.md")
WEIGHTS = os.path.join(ROOT, "models", "yolo11m.pt")


def frames():
    out = []
    for _, a, b in half_video_blocks(GAME):
        out += list(np.linspace(a, b, N_PER_HALF, dtype=int))
    cap = cv2.VideoCapture(asset_paths()["video"])
    imgs = []
    for f in out:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        imgs.append(cap.read()[1])
    return out, imgs


def run(model, imgs, device, half):
    res = []
    for img in imgs:
        r = model.predict(img, imgsz=1920, conf=0.10, classes=[0, 32], half=half, device=device, verbose=False)[0]
        res.append(np.c_[r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xywh.cpu().numpy()])
    return res


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[2] - a[4] / 2, a[3] - a[5] / 2, a[2] + a[4] / 2, a[3] + a[5] / 2
    bx0, by0, bx1, by1 = b[:, 2] - b[:, 4] / 2, b[:, 3] - b[:, 5] / 2, b[:, 2] + b[:, 4] / 2, b[:, 3] + b[:, 5] / 2
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    return inter / (a[4] * a[5] + b[:, 4] * b[:, 5] - inter)


def compare(ref, other):
    dcount, dpos, dconf, ball_same, unmatched = 0, 0.0, 0.0, 0, 0
    for r, o in zip(ref, other):
        dcount += abs(len(r) - len(o))
        for box in r:
            same = o[o[:, 0] == box[0]]
            if not len(same):
                unmatched += 1
                continue
            j = np.argmax(iou(box, same))
            if iou(box, same)[j] < 0.5:
                unmatched += 1
                continue
            dpos = max(dpos, float(np.hypot(*(box[2:4] - same[j, 2:4]))))
            dconf = max(dconf, float(abs(box[1] - same[j, 1])))
        rb = {tuple(np.round(b[2:4]).astype(int)) for b in r if b[0] == 32 and b[1] >= 0.3}
        ob = {tuple(np.round(b[2:4]).astype(int)) for b in o if b[0] == 32 and b[1] >= 0.3}
        ball_same += rb == ob
    return dcount, unmatched, dpos, dconf, ball_same / len(ref)


def main():
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    LOGGER.setLevel("ERROR")
    fs, imgs = frames()
    model = YOLO(WEIGHTS)
    modes = [("GPU fp16 (a)", 0, True), ("GPU fp16 (b)", 0, True), ("GPU fp32", 0, False), ("CPU fp32", "cpu", False)]
    runs = {}
    for name, dev, half in modes:
        t = time.time()
        runs[name] = run(model, imgs, dev, half)
        print(name, "{:.0f} s".format(time.time() - t), flush=True)
    ref = runs["GPU fp16 (a)"]
    pairs = [("GPU fp16 (a)", n) for n in runs if n != "GPU fp16 (a)"] + [("GPU fp32", "CPU fp32")]
    lines = ["# P3 Task 1: dedektörün belirlenimciliği", "",
             "`analysis/p3d_determinism.py`; {} kare, YOLO11m, imgsz 1920, conf ≥ 0.10. Referans: GPU fp16 (a).".format(len(fs)), "",
             "| karşılaştırma | kutu sayısı farkı (toplam) | eşleşmeyen kutu | en büyük merkez farkı px | en büyük güven farkı | top kümesi aynı (conf ≥ 0.3) |",
             "|---|---|---|---|---|---|"]
    for a, b in pairs:
        c = compare(runs[a], runs[b])
        lines.append("| {} ↔ {} | {} | {} | {:.3f} | {:.4f} | {:.0%} |".format(a, b, *c))
        print(a, b, c)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
