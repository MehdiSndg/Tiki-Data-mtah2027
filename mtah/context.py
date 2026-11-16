"""Per-frame context from sources other than the tracking: Smart Tagging and the referee's whistle.

    f = tag_features(load_smart_tagging.load(), n)
    w = whistle_feature(n)

Smart Tagging codes imply a side in possession: attacking codes name the team with the ball,
defending codes the team without it (code definitions in OVERVIEW.md). Several codes come as
mirrored pairs -- the same interval tagged once per team, e.g. BUILD UP for one side and
DEFENDING IN ATTACKING THIRD for the other (notes/BULGULAR.md, E) -- and a pair is one piece of
evidence, so it is counted once. It is a weak prior: ~70% agreement with possession (EDA E).

The whistle feature is the tonal-peak score at 3.6-4.0 kHz from analysis/f_audio.py (EDA F): a
soft signal, never a detector on its own.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Params
TEAM = {"Barcelona": 0, "Atletic Club": 1}
# attacking code -> its mirrored defending code (None: no mirror)
ATTACKING = {"BUILD UP": "DEFENDING IN ATTACKING THIRD", "PROGRESSION": "DEFENDING IN MIDDLE THIRD",
             "CREATING CHANCES": "DEFENDING IN DEFENSIVE THIRD", "ATTACKING TRANSITION": "DEFENSIVE TRANSITION",
             "BALL IN FINAL THIRD": None, "BALL IN THE BOX": None, "PLAYERS IN THE BOX": None,
             "LONG BALL": None}
DEFENDING = {v: k for k, v in ATTACKING.items() if v is not None}
SETPIECE_TYPES = ("none", "kickoff", "corner", "free_kick", "goal_kick", "throw_in")
_TYPE_OF = {"kick off (start)": "kickoff", "kick off (after goal)": "kickoff", "corner kick": "corner",
            "free kick": "free_kick", "goal kick": "goal_kick", "throw-in": "throw_in"}
WHISTLE_CACHE = os.path.join(ROOT, "cache", "audio_whistle.npz")
WHISTLE_BAND = (3600.0, 4000.0)
FPS = 25


def tag_features(events, n):
    """tag_H, tag_A (int8 counts of distinct tags implying that side has the ball), setpiece, setpiece_type."""
    evidence = set()
    for e in events:
        code, team = e["code"], TEAM.get(e["tags"].get("Team"))
        if team is None:
            continue
        if code in ATTACKING:
            evidence.add((code, e["start_frame"], e["end_frame"], team))
        elif code in DEFENDING:
            evidence.add((DEFENDING[code], e["start_frame"], e["end_frame"], 1 - team))
    tag = np.zeros((n, 2), np.int16)
    for _, a, b, side in evidence:
        tag[max(a, 0):min(b, n), side] += 1

    setpiece = np.zeros(n, bool)
    kind = np.zeros(n, np.int8)
    for e in events:
        if e["code"] != "SET PIECES":
            continue
        a, b = max(e["start_frame"], 0), min(e["end_frame"], n)
        setpiece[a:b] = True
        name = _TYPE_OF.get((e["tags"].get("Type") or "").strip().lower(), "none")
        kind[a:b] = SETPIECE_TYPES.index(name)
    return {"tag_H": np.minimum(tag[:, 0], 127).astype(np.int8), "tag_A": np.minimum(tag[:, 1], 127).astype(np.int8),
            "setpiece": setpiece, "setpiece_type": kind}


def whistle_feature(n):
    """(n,) float32: tonal-peak score of the audio window nearest each frame, 0 outside the whistle band."""
    if not os.path.exists(WHISTLE_CACHE):
        sys.path.insert(0, os.path.join(ROOT, "analysis"))
        import f_audio
        f_audio.compute()
    with np.load(WHISTLE_CACHE) as d:
        t, tonal, hz = d["t"], d["tonal"], d["peak_hz"]
    score = np.where((hz >= WHISTLE_BAND[0]) & (hz <= WHISTLE_BAND[1]), tonal, 0.0)
    centres = (np.arange(n) + 0.5) / FPS
    nearest = np.clip(np.searchsorted(t, centres), 1, len(t) - 1)
    nearest -= (centres - t[nearest - 1]) < (t[nearest] - centres)
    return score[nearest].astype(np.float32)
