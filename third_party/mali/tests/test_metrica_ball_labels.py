"""Metrica ball-status labels: the dead flag follows the ball channel, possession follows the events."""

import numpy as np
import pandas as pd

from mtah.constants import A, D, H, X
from mtah.io.metrica_open import ball_status_labels, derive_labels


def _events(rows):
    return pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start", "end"])


def test_dead_starts_when_the_ball_leaves_not_at_the_kick():
    frame, period = np.arange(1, 601), np.ones(600, np.int8)
    ev = _events([("Home", "PASS", "", 1, 1, 20), ("Home", "PASS", "", 1, 100, 110),
                  ("Home", "BALL OUT", "", 1, 200, 260),          # kick at 200, out at 260
                  ("Away", "SET PIECE", "THROW IN", 1, 400, 400), ("Away", "PASS", "", 1, 400, 420)])
    missing = np.zeros(600, bool)
    missing[260:399] = True                                       # frames 261..399 (index = frame - 1)
    missing[50:53] = True                                         # a tracking drop-out in live play
    lab = ball_status_labels(ev, frame, period, missing, min_dead=50)
    assert (lab[:260] == H).all()                                 # the flight out still belongs to the kicker
    assert (lab[260:399] == D).all()
    assert (lab[399:419] == A).all() and (lab[420:] == X).all()    # nothing known after the last event
    old = derive_labels(ev, frame, period)
    assert (old[199:260] == D).all()                              # the event rule called the flight dead


def test_live_play_after_an_untagged_restart_is_not_dead():
    frame, period = np.arange(1, 801), np.ones(800, np.int8)
    ev = _events([("Home", "PASS", "", 1, 1, 20), ("Away", "FAULT RECEIVED", "", 1, 100, 100),
                  ("Away", "PASS", "", 1, 180, 200), ("Away", "PASS", "", 1, 300, 320),      # restart tagged PASS
                  ("Home", "RECOVERY", "", 1, 500, 500), ("Home", "SET PIECE", "", 1, 700, 700)])
    missing = np.zeros(800, bool)
    missing[100:170] = True
    lab = ball_status_labels(ev, frame, period, missing, min_dead=50)
    assert (lab[100:170] == D).all()
    assert (lab[170:179] == X).all()                              # gap ended before the next control event
    assert (lab[179:499] == A).all() and (lab[499:699] == H).all()
    assert (derive_labels(ev, frame, period)[200:650] == D).all()  # the event rule kept it dead until SET PIECE
