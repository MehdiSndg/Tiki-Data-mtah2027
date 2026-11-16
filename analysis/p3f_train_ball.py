"""P3 Task 3b: fine-tune YOLO on the agreement-labelled tiles, then compare it with the pretrained
model on the validation blocks. GT-free.

    uv run python analysis/p3f_train_ball.py train     # ~40 min on an RTX 4060; models/ball_v1.pt
    uv run python analysis/p3f_train_ball.py eval      # -> eda_out/reports/P3_ball_detector.md

Evaluation frames: every EVAL_STEP-th frame of the validation blocks (every fifth 10-minute block,
never the public window), full 1920 x 1080 frames, fp32 (eda_out/reports/P3_determinism.md). Per model:
  ATD agreement   on frames whose ATD ball is observed and slow (< 8 m/s): recall = a box within
                  30 px; precision = the best on-pitch, off-spot box is within 30 px
  gap fill        on frames the v0 rule calls alive with the ATD ball missing: share with an
                  on-pitch, off-spot box
  spot boxes      share of frames with a box on a painted spot while the ATD ball is > 5 m away
"""

import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import asset_paths  # noqa: E402
from mtah.ball import MISSING, OBSERVED  # noqa: E402
from mtah.data import PITCH, load_tactical  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from p3a_yolo_ball import SPOTS, atd_ball_px, on_pitch, on_spot  # noqa: E402
from p3e_ball_pseudolabels import BLOCK, OUT as DATA, PUBLIC  # noqa: E402

# Params
BASE = os.path.join(ROOT, "models", "yolo11m.pt")
WEIGHTS = os.path.join(ROOT, "models", "ball_v1.pt")
RUNS = os.path.join(ROOT, "cache", "ball_runs")
EPOCHS, BATCH, FREEZE, SEED = 40, 16, 10, 0
EVAL_STEP = 10
CONFS = (0.2, 0.3, 0.5)
REPORT = os.path.join(ROOT, "eda_out", "reports", "P3_ball_detector.md")


def train():
    from ultralytics import YOLO
    model = YOLO(BASE)
    model.train(data=os.path.join(DATA, "data.yaml"), imgsz=640, epochs=EPOCHS, batch=BATCH, freeze=FREEZE,
                single_cls=True, seed=SEED, deterministic=True, workers=4, project=RUNS, name="v1",
                exist_ok=True, plots=False, verbose=False)
    shutil.copy(os.path.join(RUNS, "v1", "weights", "best.pt"), WEIGHTS)
    print("weights ->", WEIGHTS)


def detect(model, frames, cls):
    cap = cv2.VideoCapture(asset_paths()["video"])
    want, rows = set(int(f) for f in frames), []
    for f in range(int(max(frames)) + 1):
        if not cap.grab():
            break
        if f not in want:
            continue
        r = model.predict(cap.retrieve()[1], imgsz=1920, conf=0.1, classes=[cls], half=False, verbose=False)[0]
        for c, (x, y, w, h) in zip(r.boxes.conf.cpu().numpy(), r.boxes.xywh.cpu().numpy()):
            rows.append((f, c, x, y, w, h))
    return pd.DataFrame(rows, columns=["frame", "conf", "x", "y", "w", "h"])


def scores(boxes, frames, m, df, v0):
    u, v = atd_ball_px(m)
    boxes = boxes.copy()
    boxes["pitch"] = on_pitch(boxes, m["hom"]) if len(boxes) else []
    boxes["spot"] = on_spot(boxes, m["hom"]) if len(boxes) else []
    st, sp = df["ball_status"].values, df["ball_speed"].values
    seen = frames[(st[frames] == OBSERVED) & (sp[frames] < 8)]
    gaps = frames[(st[frames] == MISSING) & np.isin(v0[frames], ["H", "A"])]
    X, Y = m["bx"] * PITCH[0], m["by"] * PITCH[1]
    out = {}
    for c in CONFS:
        b = boxes[(boxes.conf >= c)]
        good = b[b.pitch & ~b.spot]
        d = np.hypot(good.x.values - u[good.frame.values], good.y.values - v[good.frame.values])
        hit = set(good.frame.values[d < 30])
        best = good.sort_values("conf").groupby("frame").tail(1)
        bd = np.hypot(best.x.values - u[best.frame.values], best.y.values - v[best.frame.values])
        bs = best[np.isin(best.frame.values, seen)]
        prec = np.mean(bd[np.isin(best.frame.values, seen)] < 30) if len(bs) else np.nan
        sf = np.unique(b[b.spot].frame.values)
        spot_d = np.min([np.hypot(X[sf] - sx, Y[sf] - sy) for sx, sy in SPOTS], axis=0) if len(sf) else np.array([])
        spot_frames = sf[spot_d > 5]                     # the ATD ball is elsewhere: the box is the paint
        out[c] = {"recall": np.mean(np.isin(seen, list(hit))), "precision": prec,
                  "gap_fill": np.mean(np.isin(gaps, good.frame.values)),
                  "spot": len(spot_frames) / max(1, len(frames))}
    return out, len(seen), len(gaps)


def evaluate():
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    LOGGER.setLevel("ERROR")
    m, df = load_tactical(), load_frames()
    v0 = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "v0_rules.csv")).label.values
    g = np.flatnonzero(m["graded"])
    val = g[((g // BLOCK) % 5 == 4) & ((g < PUBLIC[0]) | (g > PUBLIC[1]))][::EVAL_STEP]
    lines = ["# P3: kendini eğiten top dedektörü", "",
             "`analysis/p3f_train_ball.py eval`; doğrulama blokları (her 5. 10 dakikalık blok, public hariç),",
             "{} tam kare, fp32. GT kullanılmadı.".format(len(val)), "",
             "| model | conf | ATD topu recall | precision | oyun içi boşluk doldurma | nokta yanlışı (kare payı) |",
             "|---|---|---|---|---|---|"]
    for name, weights, cls in (("pretrained YOLO11m", BASE, 32), ("ince ayarlı (ball_v1)", WEIGHTS, 0)):
        boxes = detect(YOLO(weights), val, cls)
        s, n_seen, n_gap = scores(boxes, val, m, df, v0)
        for c, r in s.items():
            lines.append("| {} | {} | {:.1%} | {:.1%} | {:.1%} | {:.1%} |".format(name, c, r["recall"], r["precision"], r["gap_fill"], r["spot"]))
        print(name, s, flush=True)
    lines += ["", "ATD topunun görülüp yavaş olduğu kare: {}; oyun içi ATD boşluğu (v0'a göre): {}.".format(n_seen, n_gap)]
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    {"train": train, "eval": evaluate}[sys.argv[1]]()
