import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "starter"))

from conftest import META, needs_data, synthetic_kickoff
from mtah.periods import BREAK, FIRST, PRE2, attack_sign, detect_kickoff, periods


def test_detect_kickoff_synthetic():
    px, py, team, bx, by = synthetic_kickoff(n=300, kick_at=121, second_kick_at=250)
    assert detect_kickoff(px, py, team, bx, by, 0, 299, sign=+1) == 121


def test_detect_kickoff_none_without_shape():
    px, py, team, bx, by = synthetic_kickoff(n=300, kick_at=None)
    assert detect_kickoff(px, py, team, bx, by, 0, 299, sign=+1) is None


@needs_data
def test_attack_sign_from_metadata():
    assert attack_sign(META) == {1: +1, 2: -1}


@needs_data
def test_periods_on_match():
    from mtah.data import load_tactical
    code, kick = periods(load_tactical(), META)
    assert len(code) == 199319
    from mtah.periods import POSTH, half_of
    assert code[90651] in (FIRST, POSTH) and half_of(code)[90651] == 1 and code[90652] == BREAK   # inclusive half end
    assert abs(kick[1] - 11185) <= 25                             # published: graded from kickoff
    assert abs(kick[2] - 113742) <= 25                            # published 2nd-half kickoff
    assert (code == PRE2).sum() == kick[2] - 110808 + max(0, kick[1] - 11185)


def test_detect_kickoff_when_resting_ball_is_not_tracked():
    # The ATD drops a stationary ball: it is NaN until two frames after the strike.
    px, py, team, bx, by = synthetic_kickoff(n=300, kick_at=121)
    bx[:123] = np.nan
    by[:123] = np.nan
    assert abs(detect_kickoff(px, py, team, bx, by, 0, 299, sign=+1) - 121) <= 1


def test_detect_kickoff_ignores_parked_goalkeeper_slot():
    # A goalkeeper slot parked at a default spot in the wrong half must not veto the shape.
    px, py, team, bx, by = synthetic_kickoff(n=300, kick_at=121)
    px[:, 21] = 0.06                      # away slot sitting in front of the home goal
    assert detect_kickoff(px, py, team, bx, by, 0, 299, sign=+1, exclude=(21,)) == 121


@needs_data
def test_after_goal_kickoffs_found_when_the_ball_is_seen_at_the_spot():
    # Held-out check: no parameter was chosen on these four kickoffs. detect_kickoff's contract
    # needs the ball to be seen near the spot; for every tagged "Kick Off (After Goal)" where it
    # is (within 10 s of the window), the strike must be found inside the window.
    import load_smart_tagging as st
    from mtah.data import GK_SLOTS, load_tactical
    from mtah.periods import K2_BALL, PITCH
    m = load_tactical()
    sign = attack_sign(META)
    spot = np.hypot((m["bx"] - 0.5) * PITCH[0], (m["by"] - 0.5) * PITCH[1])
    windows = [(e["start_frame"], e["end_frame"]) for e in st.load()
               if (e["tags"].get("Type") or "").lower() == "kick off (after goal)"]
    assert len(windows) == 4
    observable = 0
    for s, e in windows:
        if not np.any(spot[s - 250:e + 250] <= K2_BALL):
            continue                      # ball never tracked near the spot: out of contract
        observable += 1
        half = 1 if s < 100000 else 2
        k = detect_kickoff(m["px"], m["py"], m["slot_team"], m["bx"], m["by"], s - 250, e, sign[half],
                           exclude=tuple(GK_SLOTS.values()))
        assert k is not None and s <= k <= e, (s, e, k)
    assert observable >= 3


def test_search_is_bounded_to_the_start_of_the_half():
    # No kickoff shape at the half start; an after-goal kickoff forms long after. It is not the
    # half's kickoff, and must not be returned (review finding 1).
    from mtah.periods import SEARCH_FRAMES
    n = SEARCH_FRAMES + 600
    px, py, team, bx, by = synthetic_kickoff(n=n, kick_at=SEARCH_FRAMES + 300, shape_from=SEARCH_FRAMES + 100)
    assert detect_kickoff(px, py, team, bx, by, 0, n - 1, sign=+1) is None


def test_kickoff_never_returned_before_the_search_start():
    # A strike extrapolated back from a late first sighting must not land before `first`
    # (review finding 3).
    px, py, team, bx, by = synthetic_kickoff(n=300, kick_at=121)
    bx[:123] = np.nan
    by[:123] = np.nan
    assert detect_kickoff(px, py, team, bx, by, 123, 299, sign=+1) == 123


def test_tracked_ball_is_not_back_extrapolated():
    # The ball rests 3.3 m from the spot, tracked throughout, and moves off radially at frame 121.
    # Back-extrapolating it as if it had been unseen would put the strike 8 frames early
    # (review finding 2).
    px, py, team, bx, by = synthetic_kickoff(n=300, kick_at=121)
    off = 3.3 / 105.3
    bx[:121] = 0.5 - off
    bx[121:] = 0.5 - off - np.arange(1, 300 - 121 + 1) * 0.45 / 105.3
    assert detect_kickoff(px, py, team, bx, by, 0, 299, sign=+1) == 121


@needs_data
def test_unfound_kickoff_makes_the_whole_half_play(monkeypatch):
    # Pins the documented fallback: no kickoff found -> no PRE2 in that half.
    import mtah.periods as P
    from mtah.data import load_tactical
    monkeypatch.setattr(P, "detect_kickoff", lambda *a, **k: None)
    code, kick = P.periods(load_tactical(), META)
    assert kick == {1: None, 2: None} and not (code == PRE2).any()
    assert (code[110808:190198] == P.SECOND).all()


def test_half_end_at_the_last_strong_whistle_before_the_players_leave():
    from mtah.periods import detect_half_end
    n = 4000
    players = np.full(n, 20.0)
    players[3000:] = 2.0                                   # the players leave for good at 3000
    w = np.zeros(n)
    w[1500] = 5000                                         # an earlier, louder stoppage whistle
    w[2950], w[2940] = 2000, 400                           # the final whistle, its burst starts at 2940
    assert detect_half_end(players, w, 0, n - 1) == 2940


def test_half_end_none_while_the_players_stay():
    from mtah.periods import detect_half_end
    n = 4000
    players = np.full(n, 20.0)
    players[1000:1500] = 2.0                               # a celebration off camera, then play again
    w = np.zeros(n)
    w[990] = 3000
    assert detect_half_end(players, w, 0, n - 1) is None


def test_half_end_at_the_collapse_without_a_whistle():
    from mtah.periods import detect_half_end
    n = 4000
    players = np.full(n, 20.0)
    players[3000:] = 1.0
    assert detect_half_end(players, np.zeros(n), 0, n - 1) == 3000


def test_half_end_ignores_a_last_second_dip():
    from mtah.periods import detect_half_end
    n = 4000
    players = np.full(n, 20.0)
    players[-30:] = 0.0                                    # the recording stops just after the whistle
    assert detect_half_end(players, np.zeros(n), 0, n - 1) is None


@needs_data
def test_first_half_ends_at_the_final_whistle_and_the_second_at_its_block_end():
    from mtah.data import load_tactical
    from mtah.periods import POSTH
    code, _ = periods(load_tactical(), META)
    post = np.flatnonzero(code == POSTH)
    assert 88670 <= post.min() <= 88710 and post.max() == 90651     # 59:07-59:08, the user's reading
    assert not np.any(post > 100000)                                # the second half plays to its end
