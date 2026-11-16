"""Dead-ball candidates and restart types from the tracking alone, against Smart Tagging. No GT.

    uv run python analysis/s_setpieces.py

Candidate dead-ball stretch: a run of MISSING ball (P1 status) of at least MIN_GAP seconds -- in
Metrica's own sample tracking such runs are dead ball almost always (analysis/g2_metrica_events.py).
The restart that ends it is typed from the players in its last LOOK seconds, with the
set-piece triggers and tolerances of Vidal-Codina et al. (2022, Fig. 4-5, S4), in their
hierarchy kickoff > corner > throw-in > goal kick > free kick (penalties are not looked for):

  corner     a player within c = 3 m of a corner mark
  throw-in   a player beyond the sideline minus t = 0.5 m, away from the corners
  goal kick  a goalkeeper inside his goal area (+ g = 0.5 m)
  free kick  none of the above

The restarting team is the team of the triggering player (for a goal kick, the goalkeeper's).
Written to cache/restarts_v1.csv for later sections.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging as st  # noqa: E402
from games import asset_paths  # noqa: E402
from mtah.ball import MISSING  # noqa: E402
from mtah.data import FPS, PITCH, load_tactical  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.periods import attack_sign, half_of, periods  # noqa: E402
from mtah.players import team_arrays  # noqa: E402

# Params
MIN_GAP = 2.0                # s
LOOK = 3.0                   # s before the end of the gap
C, T, G = 3.0, 0.5, 0.5      # m, Vidal-Codina S4
GOAL_AREA = (5.5, 9.16)      # m: depth, half-width of the goal area
HALF_L, HALF_W = PITCH[0] / 2, PITCH[1] / 2
OUT = os.path.join(ROOT, "cache", "restarts_v1.csv")
TYPE_OF_TAG = {"corner kick": "corner", "throw-in": "throw_in", "goal kick": "goal_kick", "free kick": "free_kick",
               "kick off (after goal)": "kickoff", "kick off (start)": "kickoff"}


def runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def classify(t, a, b):
    """(type, team or None, share of look-window frames with the trigger) for frames [a, b)."""
    votes = {}
    for team, side in ((0, "H"), (1, "A")):
        xy = t[side + "_xy"][a:b]
        X, Y = xy[..., 0], xy[..., 1]
        with np.errstate(invalid="ignore"):
            corner = np.any(np.hypot(HALF_L - np.abs(X), HALF_W - np.abs(Y)) <= C, axis=1)
            throw = np.any((np.abs(Y) >= HALF_W - T) & (np.abs(X) <= HALF_L - C), axis=1)
            gk = t[side + "_gk"][a:b]
            # in the attack frame every team defends -x... after normalisation H defends -x, A +x
            own_goal_x = -HALF_L if team == 0 else HALF_L
            gkick = (np.abs(gk[:, 0] - own_goal_x) <= GOAL_AREA[0] + G) & (np.abs(gk[:, 1]) <= GOAL_AREA[1] + G)
        for name, flag in (("corner", corner), ("throw_in", throw), ("goal_kick", gkick)):
            share = float(np.mean(flag)) if len(flag) else 0.0
            if share > votes.get(name, (0, None))[0]:
                votes[name] = (share, team)
    for name in ("corner", "throw_in", "goal_kick"):          # Vidal-Codina hierarchy
        if name in votes and votes[name][0] >= 0.5:
            return name, votes[name][1], votes[name][0]
    return "free_kick", None, 0.0


def classify_by_reappearance(df, t, b):
    """(type, team) from where the ball is first seen again after the gap and who is nearest it.

    The ATD drops a resting ball and picks it up once it is played, so the first sighting after a
    gap marks the restart spot, and the player nearest it the taker. Thresholds as above.
    """
    after = np.flatnonzero(df["ball_status"].values[b:b + 2 * FPS] <= 1)
    if not len(after):
        return "unknown", ""
    f = b + int(after[0])
    X, Y = float(df["ball_X"].values[f]), float(df["ball_Y"].values[f])
    near = {side: np.nanmin(np.hypot(t[side + "_xy"][f, :, 0] - X, t[side + "_xy"][f, :, 1] - Y), initial=np.inf)
            for side in ("H", "A")}
    for side in ("H", "A"):
        gk = t[side + "_gk"][f]
        near[side] = min(near[side], float(np.hypot(gk[0] - X, gk[1] - Y)) if not np.isnan(gk[0]) else np.inf)
    taker = min(near, key=near.get) if np.isfinite(min(near.values())) else ""
    if np.hypot(X, Y) <= 4.0:
        return "kickoff", taker
    if np.hypot(HALF_L - abs(X), HALF_W - abs(Y)) <= C + 2:
        return "corner", taker
    if abs(X) >= HALF_L - GOAL_AREA[0] - G - 2 and abs(Y) <= GOAL_AREA[1] + G + 2:
        return "goal_kick", taker
    if abs(Y) >= HALF_W - T - 3:
        return "throw_in", taker
    return "free_kick", taker


def main():
    meta = asset_paths()["atd_meta"]
    m = load_tactical()
    code, _ = periods(m, meta)
    half = half_of(code)
    t = team_arrays(m, half, attack_sign(meta))
    df = load_frames()
    gaps = [(a, b) for a, b in runs(df["ball_status"].values == MISSING) if b - a >= MIN_GAP * FPS]
    rows = []
    for a, b in gaps:
        typ, team, share = classify(t, max(a, b - int(LOOK * FPS)), b)
        rtyp, rteam = classify_by_reappearance(df, t, b)
        rows.append({"start": a, "end": b, "dur_s": (b - a) / FPS, "trig_type": typ,
                     "trig_team": {0: "H", 1: "A", None: ""}[team], "share": share,
                     "type": rtyp, "team": rteam,
                     "med_speed": float(np.nanmedian(df["med_speed"].values[a:b]))})
    r = pd.DataFrame(rows)
    r.to_csv(OUT, index=False)
    print("candidate dead-ball stretches (ball missing >= {:.0f} s): {}; duration s p25 {:.0f} p50 {:.0f} p75 {:.0f}".format(
        MIN_GAP, len(r), *np.percentile(r["dur_s"], [25, 50, 75])))
    print("restart type, player triggers:", r["trig_type"].value_counts().to_dict())
    print("restart type, ball reappearance:", r["type"].value_counts().to_dict())
    print("player median speed inside the stretches: p50 {:.2f} m/s (whole graded match: {:.2f})".format(
        r["med_speed"].median(), np.nanmedian(df["med_speed"].values[m["graded"]])))

    tags = [e for e in st.load() if e["code"] == "SET PIECES"]
    team_of = {"Barcelona": "H", "Atletic Club": "A"}
    matched, type_ok, team_ok, team_n, trig_type_ok, trig_team_ok = 0, 0, 0, 0, 0, 0
    for e in tags:
        typ = TYPE_OF_TAG.get((e["tags"].get("Type") or "").lower())
        hit = r[(r["start"] <= e["end_frame"] + 5 * FPS) & (r["end"] >= e["start_frame"] - 5 * FPS)]
        if len(hit):
            matched += 1
            g = hit.iloc[0]
            type_ok += g["type"] == typ
            trig_type_ok += g["trig_type"] == typ
            if team_of.get(e["tags"].get("Team")):
                team_n += 1
                team_ok += g["team"] == team_of[e["tags"]["Team"]]
                trig_team_ok += g["trig_team"] == team_of[e["tags"]["Team"]]
    print("\nSmart Tagging set pieces: {}; overlapped by a candidate stretch (+/-5 s): {} ({:.0%})".format(
        len(tags), matched, matched / len(tags)))
    print("   same restart type: triggers {}/{}, reappearance {}/{}".format(trig_type_ok, matched, type_ok, matched))
    print("   same team (where the tag names one): triggers {}/{}, reappearance {}/{}".format(trig_team_ok, team_n, team_ok, team_n))
    covered = np.zeros(len(r), bool)
    for e in tags:
        covered |= ((r["start"] <= e["end_frame"] + 5 * FPS) & (r["end"] >= e["start_frame"] - 5 * FPS)).values
    print("candidate stretches with no Smart Tagging set piece nearby: {} of {}; their types: {}".format(
        int((~covered).sum()), len(r), r.loc[~covered, "type"].value_counts().to_dict()))


if __name__ == "__main__":
    main()
