"""Match constants for BAR-ATH. Mirrors more-than-a-hack-2027/games.py (checked by tests)."""

import numpy as np

GAME = "BAR-ATH"
N_FRAMES = 199_319                      # tactical video frames == submission rows
FPS = 25
PITCH = (105.3, 68.0)                   # metres, measured from the corrected tracking

# Graded runs, half-open video-frame intervals (the scorer derives the same from the GT).
HALVES = {1: (11_185, 90_652), 2: (110_808, 190_198)}
H2_KICKOFF = 113_742                    # [110808, 113742) is graded and entirely dead ball
FORCED_DEAD = [(110_808, 113_742)]
PUBLIC_WINDOW = (11_185, 18_685)

LABELS = ("H", "A", "D", "X")
H, A, D, X = 0, 1, 2, 3

N_SLOTS = 46                            # tactical ATD track slots: 0-23 home, 24-45 away
GK_SLOTS = (23, 45)


def graded_mask(n=N_FRAMES):
    mask = np.zeros(n, dtype=bool)
    for start, stop in HALVES.values():
        mask[start:stop] = True
    return mask


def period_array(n=N_FRAMES):
    """1 / 2 on the graded halves, 0 elsewhere (pre-match, half-time, post-match)."""
    period = np.zeros(n, dtype=np.int8)
    for half, (start, stop) in HALVES.items():
        period[start:stop] = half
    return period
