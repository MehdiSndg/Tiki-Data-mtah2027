import numpy as np
import pytest

from conftest import META, needs_data
from mtah.periods import attack_sign, half_of, periods
from mtah.players import team_arrays, to_attack_frame, view_box


def test_attack_frame_flips_second_half():
    x = np.array([0.9, 0.9]); y = np.array([0.5, 0.5]); half = np.array([1, 2])
    X, Y = to_attack_frame(x, y, half, {1: +1, 2: -1})
    assert X[0] == pytest.approx(0.4 * 105.3) and X[1] == pytest.approx(-0.4 * 105.3)


def test_attack_frame_nan_outside_halves():
    X, Y = to_attack_frame(np.array([0.3]), np.array([0.3]), np.array([0]), {1: +1, 2: -1})
    assert np.isnan(X[0]) and np.isnan(Y[0])


def test_half_of_assigns_pre2_to_its_half():
    from mtah.periods import BREAK, FIRST, PRE, PRE2, SECOND, POST
    code = np.array([PRE, PRE2, FIRST, BREAK, PRE2, SECOND, POST], np.int8)
    assert half_of(code).tolist() == [0, 1, 1, 0, 2, 2, 0]


@pytest.fixture(scope="module")
def match():
    from mtah.data import load_tactical
    m = load_tactical()
    code, _ = periods(m, META)
    return m, half_of(code), team_arrays(m, half_of(code), attack_sign(META))


@needs_data
def test_goalkeepers_defend_minus_x_after_normalisation(match):
    m, half, t = match
    for h in (1, 2):
        sel = half == h
        assert np.nanmean(t["H_gk"][sel, 0]) < -30 and np.nanmean(t["A_gk"][sel, 0]) > 30


@needs_data
def test_empty_slots_dropped(match):
    m, half, t = match
    assert t["H_xy"].shape[1] == 21 and t["A_xy"].shape[1] == 20


@needs_data
def test_view_box_contains_ball_when_visible(match):
    m, half, t = match
    vb = view_box(m["hom"])
    ok = ~np.isnan(m["bx"]) & m["graded"] & (m["bx"] >= 0) & (m["bx"] <= 1)
    inside = (vb[ok, 0] <= m["bx"][ok]) & (m["bx"][ok] <= vb[ok, 1])
    assert inside.mean() > 0.97


def test_half_of_assigns_the_post_whistle_stretch_to_its_half():
    from mtah.periods import BREAK, FIRST, POST, POSTH, PRE2, SECOND, half_of
    code = np.array([FIRST, POSTH, BREAK, PRE2, SECOND, POSTH, POST], np.int8)
    assert half_of(code).tolist() == [1, 1, 0, 2, 2, 2, 0]
