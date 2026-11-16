import numpy as np
import pytest

from conftest import META, first_code, needs_data, straight_line
from mtah.ball import INTERP, MASKED, MISSING, OBSERVED, OUTLIER, clean_ball
from mtah.periods import BREAK, FIRST, attack_sign, half_of, periods

ONE = {1: +1}


def test_masked_outside_play():
    bx, by = straight_line(10.0, 100)
    code = np.r_[np.full(10, BREAK), np.full(90, FIRST)].astype(np.int8)
    out = clean_ball(bx, by, code, np.ones(100), ONE)
    assert (out["status"][:10] == MASKED).all() and (out["status"][10:] == OBSERVED).all()


def test_short_gap_interpolated_long_gap_missing():
    bx, by = straight_line(10.0, 120, gaps=[(30, 35), (70, 76)])     # 5 and 6 frames
    out = clean_ball(bx, by, first_code(120), np.ones(120), ONE)
    assert (out["status"][30:35] == INTERP).all() and (out["status"][70:76] == MISSING).all()


def test_smoothing_keeps_gap_edges():
    bx, by = straight_line(10.0, 120, gaps=[(70, 76)])
    out = clean_ball(bx, by, first_code(120), np.ones(120), ONE)
    assert not np.isnan(out["X"][69]) and not np.isnan(out["X"][76])


def test_speed_of_constant_motion():
    bx, by = straight_line(10.0, 50)
    out = clean_ball(bx, by, first_code(50), np.ones(50), ONE)
    assert np.nanmedian(out["speed"]) == pytest.approx(10.0, rel=0.02)


def test_outlier_run_flagged():
    bx, by = straight_line(10.0, 100, excursion=(40, 43))
    out = clean_ball(bx, by, first_code(100), np.ones(100), ONE)
    assert (out["status"][40:43] == OUTLIER).all() and out["status"][43] == OBSERVED


def test_stationary_flag():
    bx, by = straight_line(0.0, 40)
    out = clean_ball(bx, by, first_code(40), np.ones(40), ONE)
    assert out["stationary"][10:30].all()


def test_gap_age_and_length():
    bx, by = straight_line(10.0, 60, gaps=[(20, 30)])
    out = clean_ball(bx, by, first_code(60), np.ones(60), ONE)
    assert out["gap_age"][19] == 0 and out["gap_age"][25] == 6 and out["gap_len"][25] == 10


@needs_data
def test_warmup_ball_masked_on_match():
    from mtah.data import load_tactical
    m = load_tactical()
    code, kick = periods(m, META)
    out = clean_ball(m["bx"], m["by"], code, half_of(code), attack_sign(META))
    assert (out["status"][110808:kick[2]] == MASKED).all()
