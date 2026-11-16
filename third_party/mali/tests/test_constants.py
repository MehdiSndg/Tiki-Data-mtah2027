"""Our constants must agree with the official games.py and the public ground truth."""

import sys

import numpy as np

from mtah import constants as C
from mtah.paths import OFFICIAL

sys.path.insert(0, str(OFFICIAL))
import games  # noqa: E402


def test_frames_and_halves_match_games_py():
    assert games.GAMES["BAR-ATH"]["n_video_frames"] == C.N_FRAMES
    assert games.GAMES["BAR-ATH"]["fps"] == C.FPS
    assert tuple(games.GAMES["BAR-ATH"]["pitch"]) == C.PITCH
    blocks = games.half_video_blocks("BAR-ATH")
    assert [(a, b + 1) for _, a, b in blocks] == [C.HALVES[1], C.HALVES[2]]


def test_public_window_matches_gt():
    from mtah.io.gt import public_gt
    gt = public_gt()
    known = np.flatnonzero(gt != C.X)
    assert (known[0], known[-1] + 1) == C.PUBLIC_WINDOW
    assert len(gt) == C.N_FRAMES
