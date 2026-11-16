"""Inventory E12: does the ATD drop a ball that stands still? Re-measured with ball_v1 over the whole match.

    uv run python analysis/p5k_still_ball.py [pass dir]      # default cache/ball_v1_pass

EDA B measured it with the off-the-shelf YOLO on 400 + 400 sampled frames: of the balls YOLO found
in both f and f + 10 (0.4 s), 88% were slower than 0.5 m/s where the ATD had no ball, 5% where it
had one. Same measure here with the self-trained detector (conf >= CONF, not a painted spot), on
every graded frame. The same selection bias applies: a still ball is not blurred, so it is easier
to find.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from mtah.data import PITCH  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.video_match import _hom, detector_ball  # noqa: E402

STEP, CONF, SLOW = 10, 0.5, 0.5


def main():
    det_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "cache", "ball_v1_pass")
    m = from_tactical()
    n = len(m["bx"])
    xy, conf, spot = detector_ball(_hom(), n, det_dir, 0, conf_min=CONF)
    ok = ~np.isnan(xy[:, 0]) & ~spot & (m["half"] > 0)
    f = np.flatnonzero(ok[:-STEP] & ok[STEP:] & (m["half"][:-STEP] == m["half"][STEP:]))
    L, W = PITCH
    speed = np.hypot((xy[f + STEP, 0] - xy[f, 0]) * L, (xy[f + STEP, 1] - xy[f, 1]) * W) / (STEP / 25)
    atd = ~np.isnan(m["bx"][f])
    print("| ATD'de | kare (dedektör f ve f+10'da) | top hızı medyanı m/s | < {} m/s |".format(SLOW))
    print("|---|---|---|---|")
    for name, mask in (("top yok", ~atd), ("top var", atd)):
        print("| {} | {} | {:.2f} | {:.0%} |".format(name, mask.sum(), np.median(speed[mask]), np.mean(speed[mask] < SLOW)))


if __name__ == "__main__":
    main()
