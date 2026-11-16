"""Match periods per video frame, with each half's kickoff found by an algorithm.

    from mtah.periods import periods, attack_sign
    code, kick = periods(load_tactical(), asset_paths()["atd_meta"])

Half boundaries come from games.half_video_blocks -- published in DATA.md as "boundaries, not
answers". Kickoff frames are NOT copied from the documentation: they are detected from the
tracking with the kickoff trigger of Vidal-Codina et al. (2022, sec. 2.3.4, tolerances from their
Online Resource S4), so the same code works on the next match. A graded stretch before the
detected kickoff is its own period (PRE2): play has not started there, whatever the ATD shows.
The end of a half is found the same way (POSTH): the published blocks are the tracking recording,
not the whistles, and the first half's block runs ~78 s past its final whistle
(notes/DURUM.md, item 7). Both are graded and dead: NOT_PLAYED.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from mtah.data import GK_SLOTS  # noqa: E402

# Params
PRE, FIRST, BREAK, PRE2, SECOND, POST, POSTH = 0, 1, 2, 3, 4, 5, 6
NOT_PLAYED = (PRE2, POSTH)          # graded frames of a half with no play in them: dead ball
PITCH = (105.3, 68.0)        # games.py, measured from the corrected tracking
K1 = 3.0                     # m: tolerance on "every player in his own half" (Vidal-Codina S4: 3-5)
# m: player distance to the centre spot. Vidal-Codina S4 found every k2 >= 2 m detected all
# kickoffs and advise wider tolerances for lower-quality tracking; ours is automatic.
K2 = 3.0
# m: ball distance to the centre spot. Separate and wider: our audit of the ATD ball against the
# video found ~6% of ball positions 1-5 m off (notes/BULGULAR.md, B-0).
K2_BALL = 4.0
# The rest are engineering choices, not literature values; each is stated in physical terms.
# outfield slots that must be visible for the shape to mean anything: 80% of the 20 outfield
# players (EDA C-3: the median frame shows 10 per team)
MIN_OUTFIELD = 16
MOVE_START = 0.1             # m per frame (2.5 m/s): the ball has started to move
MOVE_CONFIRM = 1.0           # m in 5 frames (5 m/s away from the spot): a pass, not jitter
CONFIRM_FRAMES = 5
MAX_BACKTRACK = 10           # frames (0.4 s): furthest an unseen strike is extrapolated back
# frames after the start of the graded half within which its kickoff must fall. A later hit is
# an after-goal kickoff, not the half's, and returning it would mask most of the half as PRE2.
SEARCH_FRAMES = 300 * 25
# End of a half. Physical reasoning, not fitted: while play goes on the camera shows a wide part of
# the pitch and the ATD tracks >= 10 outfield players on 95% of frames (EDA C-3); fewer than
# FEW_PLAYERS for good means the players have left. Celebrations off camera also drop the count,
# but play comes back after them, so the drop must last to the end of the block.
END_SEARCH_S = 300                 # the end is looked for in the block's last 5 minutes
FEW_PLAYERS = 6                    # outfield players tracked
COLLAPSE_SHARE = 0.95              # of the frames from the drop to the block's end
MIN_TAIL_S = 5                     # a dip in the last seconds of the recording is not an end
# The whistle that ended it: the latest of the half's loudest whistles (top 0.1%) in the 30 s
# before the drop; a stoppage whistle earlier in stoppage time can be louder (58:46 in the first
# half), which is why it is the latest, and why the drop of players, not the whistle, decides.
WHISTLE_BACK_S = 30
WHISTLE_Q = 0.999
BURST_S, BURST_FRAC = 3, 0.1       # the whistle starts where its burst first reaches 10% of the peak
FPS = 25


def attack_sign(meta_path):
    """{half: +1 or -1}: the direction along x in which the HOME side attacks."""
    root = ET.parse(meta_path).getroot()
    home = next(root.iter("Team"))
    for parameter in home.iter("ProviderParameter"):
        if parameter.findtext("Name") == "attack_direction_first_half":
            first = +1 if parameter.findtext("Value") == "left_to_right" else -1
            return {1: first, 2: -first}
    raise ValueError("{}: no attack_direction_first_half for the home team".format(meta_path))


def detect_kickoff(px, py, slot_team, bx, by, first, last, sign, exclude=()):
    """First kickoff frame in [first, last], or None.

    px, py (n, slots) and bx, by (n,) are normalised 0-1; slot_team is 0 home / 1 away; sign is
    the home side's attack direction in this half; `exclude` are slots left out of the own-half
    check (the goalkeepers, whose ATD slots can sit parked at a default position).

    A frame t qualifies when the ball is seen within K2_BALL of the spot and is MOVE_CONFIRM further
    away 5 frames later, while on frame t-1 the players hold the kickoff shape. The ATD drops a
    resting ball, so the ball is often first seen already in flight: only then is the strike
    extrapolated back, from its distance to the spot and its radial speed, never past the frame
    on which the ball was last seen. A ball tracked through the strike is not extrapolated. The
    frame returned is the first frame on which the ball has left its resting place, never earlier
    than `first`; only the first SEARCH_FRAMES of the range are searched.
    """
    stop = min(last + 1, first + SEARCH_FRAMES, len(bx) - CONFIRM_FRAMES - 1)
    frames = np.arange(max(first, 1), stop)
    if not len(frames):
        return None
    prev = frames - 1

    # metres from the halfway line, positive towards the side the home team attacks
    along = (px[prev] - 0.5) * PITCH[0] * sign
    outfield = np.ones(px.shape[1], bool)
    outfield[list(exclude)] = False
    home = slot_team == 0
    present = ~np.isnan(px[prev])
    with np.errstate(invalid="ignore"):
        wrong_half = np.where(home, along > K1, along < -K1) & present & outfield
    in_halves = ~np.any(wrong_half, axis=1) & ((present & outfield).sum(1) >= MIN_OUTFIELD)
    with np.errstate(invalid="ignore"):
        player_at_spot = np.any(np.hypot((px[prev] - 0.5) * PITCH[0], (py[prev] - 0.5) * PITCH[1]) <= K2, axis=1)

    spot = np.hypot((bx - 0.5) * PITCH[0], (by - 0.5) * PITCH[1])
    with np.errstate(invalid="ignore"):
        leaving = (spot[frames] <= K2_BALL) & (spot[frames + CONFIRM_FRAMES] - spot[frames] >= MOVE_CONFIRM)
    hit = np.flatnonzero(in_halves & player_at_spot & leaving)
    if not len(hit):
        return None

    # first forward step of the ball that is a real movement, from the qualifying sighting on
    t = int(frames[hit[0]])
    step = np.hypot(np.diff(bx[t:t + CONFIRM_FRAMES + 1]) * PITCH[0], np.diff(by[t:t + CONFIRM_FRAMES + 1]) * PITCH[1])
    moving = np.flatnonzero(step > MOVE_START)
    if not len(moving):
        return None
    t_move = t + int(moving[0])
    kick = t_move + 1
    if np.isnan(bx[t_move - 1]):
        # first seen already moving: extrapolate back along the radial speed, not past the last
        # frame on which the ball was seen
        radial = spot[t_move + 1] - spot[t_move]
        if radial > MOVE_START:
            back = min(int(np.ceil(spot[t_move] / radial - 1e-9)), MAX_BACKTRACK)
            seen_before = np.flatnonzero(~np.isnan(bx[max(t_move - MAX_BACKTRACK, 0):t_move]))
            if len(seen_before):
                back = min(back, MAX_BACKTRACK - int(seen_before[-1]) - 1)
            kick = t_move + 1 - back
    return max(kick, first)


def half_of(code):
    """(n,) int8: 1 or 2 for frames of either half, PRE2 and POSTH included, and 0 elsewhere.

    PRE2 belongs to whichever half it opens and POSTH to the one it closes: the half before the
    break, or the one after it.
    """
    half = np.where(code == FIRST, 1, np.where(code == SECOND, 2, 0)).astype(np.int8)
    breaks = np.flatnonzero(code == BREAK)
    split = breaks[0] if len(breaks) else len(code)
    edge = np.flatnonzero(np.isin(code, NOT_PLAYED))
    half[edge] = np.where(edge < split, 1, 2)
    return half


def detect_half_end(players, whistle, first, last):
    """First frame after the half's final whistle, or None if play runs to `last`.

    players: (n,) outfield players tracked per frame; whistle: (n,) mtah.context.whistle_feature.
    """
    few = np.asarray(players[first:last + 1]) < FEW_PLAYERS
    lo = max(0, len(few) - END_SEARCH_S * FPS)
    tail = np.cumsum(few[::-1])[::-1] / np.arange(len(few), 0, -1)     # share of `few` from i to the end
    starts = np.flatnonzero(few & ~np.r_[False, few[:-1]])
    ok = starts[(starts >= lo) & (tail[starts] >= COLLAPSE_SHARE) & (len(few) - starts >= MIN_TAIL_S * FPS)]
    if not len(ok):
        return None
    drop = first + int(ok[0])
    w = np.asarray(whistle[first:last + 1], float)
    loud = max(float(np.quantile(w, WHISTLE_Q)), 0.0)
    a = max(first, drop - WHISTLE_BACK_S * FPS)
    peaks = np.flatnonzero(np.asarray(whistle[a:drop + 1], float) > loud)
    if not len(peaks):
        return drop
    p = a + int(peaks[-1])
    b = max(first, p - BURST_S * FPS)
    burst = np.flatnonzero(np.asarray(whistle[b:p + 1], float) >= BURST_FRAC * whistle[p])
    return b + int(burst[0])


def periods(m, meta_path, whistle=None):
    """(code, kick): (n,) int8 period code per video frame, and {half: kickoff frame or None}.

    whistle: mtah.context.whistle_feature(n), read from its cache when not given.
    """
    n = len(m["bx"])
    if whistle is None:
        from mtah.context import whistle_feature
        whistle = whistle_feature(n)
    outfield = np.ones(m["px"].shape[1], bool)
    outfield[list(GK_SLOTS.values())] = False
    players = (~np.isnan(m["px"][:, outfield])).sum(1)
    sign = attack_sign(meta_path)
    blocks = half_video_blocks(GAME)
    code = np.full(n, BREAK, np.int8)
    code[:blocks[0][1]] = PRE
    code[blocks[-1][2] + 1:] = POST
    kick = {}
    for half, first, last in blocks:
        k = detect_kickoff(m["px"], m["py"], m["slot_team"], m["bx"], m["by"], first, last, sign[half],
                           exclude=tuple(GK_SLOTS.values()))
        kick[half] = k
        start = first if k is None else k
        code[first:start] = PRE2
        code[start:last + 1] = FIRST if half == 1 else SECOND
        end = detect_half_end(players, whistle, start, last)
        if end is not None:
            code[end:last + 1] = POSTH
    return code, kick
