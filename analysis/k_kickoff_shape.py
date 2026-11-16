"""Experiment for the pending P1 decision: a kickoff rule that does not use the ball at all.

    uv run python analysis/k_kickoff_shape.py

The P1 rule needs the ATD ball near the centre spot, and on the second-half kickoff the ATD ball
never comes closer than 3.34 m -- hence the widened tolerances the user has to rule on. This rule
uses the players only, with the literature's tolerances unchanged (Vidal-Codina S4: k1 = 3 m,
k2 = 2 m): the kickoff shape (every visible outfield player in his own half, one player at the
spot) must hold for SUSTAIN seconds, and the kickoff is the first frame after such a block on which
the shape breaks. Nothing here was chosen on the six kickoffs it is evaluated against.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging as st  # noqa: E402
from games import asset_paths  # noqa: E402
from mtah.data import GK_SLOTS, PITCH, load_tactical  # noqa: E402
from mtah.periods import attack_sign, detect_kickoff  # noqa: E402

# Params
K1, K2 = 3.0, 2.0            # m, Vidal-Codina S4 (unchanged)
MIN_OUTFIELD = 16
SUSTAIN = 2.0                # s: a kickoff line-up is held for seconds, a passing configuration is not


def shape(m, sign_of_frame):
    along = (m["px"] - 0.5) * PITCH[0] * sign_of_frame[:, None]
    outfield = np.ones(m["px"].shape[1], bool)
    outfield[list(GK_SLOTS.values())] = False
    present = ~np.isnan(m["px"]) & outfield
    home = m["slot_team"] == 0
    with np.errstate(invalid="ignore"):
        wrong = np.where(home, along > K1, along < -K1) & present
        at_spot = np.any(np.hypot((m["px"] - 0.5) * PITCH[0], (m["py"] - 0.5) * PITCH[1]) <= K2, axis=1)
    return ~wrong.any(1) & (present.sum(1) >= MIN_OUTFIELD) & at_spot


def shape_kickoff(ok, first, last):
    """First frame in [first, last] that ends a block of >= SUSTAIN s of kickoff shape."""
    w = ok[first:last + 1].astype(np.int8)
    edges = np.diff(np.r_[0, w, 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a >= SUSTAIN * 25:
            return first + int(b)
    return None


def main():
    m = load_tactical()
    meta = asset_paths()["atd_meta"]
    sign = attack_sign(meta)
    sof = np.where(np.arange(len(m["bx"])) < 100000, sign[1], sign[2]).astype(float)
    ok = shape(m, sof)
    cases = [("1st half", 11185 - 250, 11185 + 250, 11185), ("2nd half", 110808, 113742 + 250, 113742)]
    for e in st.load():
        if (e["tags"].get("Type") or "").lower() == "kick off (after goal)":
            cases.append(("after goal", e["start_frame"] - 250, e["end_frame"] + 250, (e["start_frame"], e["end_frame"])))
    print("{:<11} {:>16} {:>14} {:>16}".format("kickoff", "reference", "shape rule", "ball rule (P1)"))
    for name, a, b, ref in cases:
        s = shape_kickoff(ok, a, b)
        half = 1 if a < 100000 else 2
        k = detect_kickoff(m["px"], m["py"], m["slot_team"], m["bx"], m["by"], a, b, sign[half],
                           exclude=tuple(GK_SLOTS.values()))
        refs = "{}".format(ref) if isinstance(ref, int) else "{}-{}".format(*ref)
        def rel(x):
            if x is None:
                return "none"
            return "{} ({:+d})".format(x, x - ref) if isinstance(ref, int) else "{} ({})".format(x, "in" if ref[0] <= x <= ref[1] else "out")
        print("{:<11} {:>16} {:>14} {:>16}".format(name, refs, rel(s), rel(k)))


if __name__ == "__main__":
    main()
