import numpy as np
import pandas as pd

from conftest import needs_data
from mtah.keepers import ABSENT, GHOST, OFFSCREEN, SEEN, UNKNOWN, keeper_status

PITCH = (105.0, 68.0)


def zoom_hom(n):
    """The camera shows normalised x 0.25..0.75 across 1920 px and y 0..1 down 1080 px."""
    h = np.zeros((n, 3, 3))
    h[:, 0, 0], h[:, 0, 2] = 3840.0, -960.0
    h[:, 1, 1], h[:, 2, 2] = 1080.0, 1.0
    return h


def persons(rows):
    """rows of (frame, foot u px, foot v px)."""
    a = np.array(rows, float).reshape(-1, 3)
    return pd.DataFrame({"frame": a[:, 0].astype(int), "u": a[:, 1], "v": a[:, 2]})


def test_each_status():
    n = 6
    x = np.array([0.50, 0.50, 0.90, np.nan, 0.50, 0.50], np.float32)
    y = np.full(n, 0.5, np.float32)
    # frame 0: a person at the keeper; frame 1: the nearest person 10 m away; frame 4: nobody at all
    p = persons([(0, 960.0, 540.0), (1, 960.0 + 10 / 105 * 3840, 540.0)])
    covered = np.array([0, 1, 2, 3, 4])                     # frame 5 was never run through YOLO
    s = keeper_status(x, y, zoom_hom(n), p, covered, PITCH)
    assert s.tolist() == [SEEN, GHOST, OFFSCREEN, ABSENT, GHOST, UNKNOWN]


def test_two_metre_radius():
    x = np.full(2, 0.5, np.float32)
    y = np.full(2, 0.5, np.float32)
    px_per_m = 3840 / 105
    p = persons([(0, 960 + 1.9 * px_per_m, 540.0), (1, 960 + 2.1 * px_per_m, 540.0)])
    assert keeper_status(x, y, zoom_hom(2), p, np.array([0, 1]), PITCH).tolist() == [SEEN, GHOST]


@needs_data
def test_frames_carry_the_keeper_status():
    from mtah.frames import load_frames
    df = load_frames()
    g = df["half"] > 0
    for side in ("H", "A"):
        s = df.loc[g, side + "_gk_status"]
        in_view = s.isin([SEEN, GHOST])
        ghost = (s[in_view] == GHOST).mean()
        assert 0.03 < ghost < 0.35            # 22 Oct morning, 1 frame in 10: 19.9% (H), 9.6% (A)
