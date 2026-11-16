"""EDA section A: an 8-second review clip around each ground-truth transition, tracking drawn on.

    uv run python analysis/a_clips.py [--before 4 --after 4 --scale 0.6667]

Full frame (not a crop), so the context of each change is visible. Home slots blue, away orange,
goalkeepers ringed, the ATD ball as a white ring; a bar at the top shows the ground-truth label of
the current frame, its time relative to the transition, and the baseline_smooth50 label.

Output: eda_out/A/clips/ -- clipped match video. Never commit or share outside the team.
"""

import argparse
import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from a_contact_sheets import BGR  # noqa: E402
from a_gt_transitions import LABELS, WINDOW, segments  # noqa: E402
from games import asset_paths  # noqa: E402
from mtah.data import FPS, GK_SLOTS, load_tactical, project  # noqa: E402

# Params
OUT = os.path.join(ROOT, "eda_out", "A", "clips")
BASELINE = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")
LABEL_BGR = {"H": BGR[0], "A": BGR[1], "D": (140, 140, 140), "X": (60, 60, 60)}


def draw(img, m, f):
    hom = m["hom"][f]
    for s in np.flatnonzero(~np.isnan(m["px"][f])):
        x, y = project(hom, m["px"][f, s], m["py"][f, s])
        team = int(m["slot_team"][s])
        cv2.circle(img, (int(x), int(y)), 6, BGR[team], -1)
        if s == GK_SLOTS[team]:
            cv2.circle(img, (int(x), int(y)), 12, BGR[team], 2)
    if not np.isnan(m["bx"][f]):
        x, y = project(hom, m["bx"][f], m["by"][f])
        cv2.circle(img, (int(x), int(y)), 14, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--before", type=float, default=4.0)
    parser.add_argument("--after", type=float, default=4.0)
    parser.add_argument("--scale", type=float, default=2 / 3.0, help="output size relative to 1920x1080")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    m = load_tactical()
    gt = m["gt_public"]
    base = pd.read_csv(BASELINE)["label"].values
    segs = segments(gt, *WINDOW)
    width, height = int(1920 * args.scale) // 2 * 2, int(1080 * args.scale) // 2 * 2
    cap = cv2.VideoCapture(asset_paths()["video"])

    for i in range(1, len(segs)):
        f0 = segs[i][0]
        name = "t{:02d}_{}_{}to{}.mp4".format(i, f0, LABELS[segs[i - 1][2]], LABELS[segs[i][2]])
        path = os.path.join(OUT, name)
        writer = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", "{}x{}".format(width, height), "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)
        first = int(f0 - args.before * FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for f in range(first, int(f0 + args.after * FPS)):
            ok, img = cap.read()
            if not ok:
                break
            draw(img, m, f)
            label = LABELS[gt[f]]
            cv2.rectangle(img, (0, 1030), (1920, 1080), LABEL_BGR[label], -1)
            cv2.putText(img, "GT {}   {:+.2f} s   frame {}   baseline_smooth50 {}   #{}".format(
                label, (f - f0) / FPS, f, base[f], i), (20, 1066), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 2, cv2.LINE_AA)
            writer.stdin.write(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA).tobytes())
        writer.stdin.close()
        writer.wait()
        print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
