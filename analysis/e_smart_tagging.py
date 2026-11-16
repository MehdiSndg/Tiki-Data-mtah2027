"""EDA section E: how much does Smart Tagging say about who has the ball, and when play restarts?

    uv run python analysis/e_smart_tagging.py

A code implies a side in possession: attacking codes name the team with the ball, defending codes
name the team without it. That mapping is ours (from the code definitions in OVERVIEW.md), not
Metrica's, and is written down here so it can be argued with.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging as st  # noqa: E402
from mtah.data import FPS, PITCH, load_tactical  # noqa: E402

# Params
TEAM = {"Barcelona": 0, "Atletic Club": 1}
ATTACKING = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX",
             "PLAYERS IN THE BOX", "ATTACKING TRANSITION", "LONG BALL"}
DEFENDING = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
             "DEFENSIVE TRANSITION"}
PUBLIC = slice(11185, 18685)


def nearest_team(m):
    """Team of the slot nearest the ball per frame; -1 where there is no ball."""
    d = np.hypot((m["px"] - m["bx"][:, None]) * PITCH[0], (m["py"] - m["by"][:, None]) * PITCH[1])
    out = np.full(len(d), -1, np.int8)
    ok = ~np.all(np.isnan(d), axis=1)
    out[ok] = m["slot_team"][np.nanargmin(d[ok], axis=1)]
    return out


def main():
    m = load_tactical()
    n, g = len(m["bx"]), m["graded"]
    events = st.load()
    votes = np.zeros((n, 2), np.int16)            # per frame: tags implying H, tags implying A
    for e in events:
        team = TEAM.get(e["tags"].get("Team"))
        if team is None or e["code"] not in ATTACKING | DEFENDING:
            continue
        side = team if e["code"] in ATTACKING else 1 - team
        votes[e["start_frame"]:e["end_frame"], side] += 1

    any_vote = votes.sum(1) > 0
    conflict = (votes[:, 0] > 0) & (votes[:, 1] > 0)
    implied = np.where(votes[:, 0] > votes[:, 1], 0, np.where(votes[:, 1] > votes[:, 0], 1, -1))
    print("graded frames covered by a team-implying tag: {:.1%}; of those, conflicting (both sides): {:.1%}"
          .format(any_vote[g].mean(), conflict[g & any_vote].mean()))

    near = nearest_team(m)
    both = g & (implied >= 0) & (near >= 0)
    print("agreement with the nearest-player team (whole match, GT-free, ball visible): {:.1%} on {} frames"
          .format((implied[both] == near[both]).mean(), int(both.sum())))
    for code in sorted(ATTACKING | DEFENDING):
        v = np.zeros(n, np.int8) - 1
        for e in events:
            team = TEAM.get(e["tags"].get("Team"))
            if e["code"] == code and team is not None:
                v[e["start_frame"]:e["end_frame"]] = team if code in ATTACKING else 1 - team
        mk = g & (v >= 0) & (near >= 0)
        if mk.sum():
            print("   {:<30} frames {:6d}  agrees with nearest {:.1%}".format(code, int(mk.sum()), (v[mk] == near[mk]).mean()))

    gt = m["gt_public"]
    pub = np.zeros(n, bool)
    pub[PUBLIC] = True
    a = pub & (implied >= 0) & (gt < 2)
    print("\npublic window (understanding only): tag-implied side vs GT on GT-alive frames: {:.1%} on {} frames; "
          "GT-dead frames under a team tag: {:.1%}".format((implied[a] == gt[a]).mean(), int(a.sum()),
                                                        (gt[pub & (implied >= 0)] == 2).mean()))

    print("\nSET PIECES: where do they sit relative to the ball and the players?")
    with np.errstate(all="ignore"):
        v = np.hypot(np.diff(m["px"], axis=0) * PITCH[0], np.diff(m["py"], axis=0) * PITCH[1]) * FPS
        speed = np.r_[np.nanmedian(v[:1], axis=1), np.nanmedian(v, axis=1)]
    rows = []
    for e in events:
        if e["code"] != "SET PIECES":
            continue
        s, t = e["start_frame"], e["end_frame"]
        rows.append((e["tags"].get("Type"), e["tags"].get("Team"), (t - s) / FPS, np.nanmedian(speed[s:t]),
                     np.isnan(m["bx"][s:t]).mean()))
    types = sorted({r[0] for r in rows}, key=str)
    for ty in types:
        r = [x for x in rows if x[0] == ty]
        print("   {:<24} n={:2d}  duration s median {:5.1f}  player speed median {:.2f}  ball missing {:.0%}  teams {}".format(
            str(ty), len(r), np.median([x[2] for x in r]), np.nanmedian([x[3] for x in r]),
            np.mean([x[4] for x in r]), sorted({str(x[1]) for x in r})))


if __name__ == "__main__":
    main()
