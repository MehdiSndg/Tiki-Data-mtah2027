"""EDA section A: what does the tracking show around each ground-truth transition?

    uv run python analysis/a_gt_transitions.py

Understanding only. The public window is used to learn what the labels *mean*; nothing here
chooses a parameter. Writes eda_out/A/transitions.csv and prints a per-transition table.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from mtah.data import FPS, PITCH, load_tactical  # noqa: E402

# Params
OUT = os.path.join(ROOT, "eda_out", "A")
LABELS = "HADX"
WINDOW = (11185, 18685)          # the public window, half-open
CONTEXT_S = 3.0                  # seconds either side of a transition to summarise


def segments(codes, start, stop):
    w = codes[start:stop]
    cuts = np.flatnonzero(w[1:] != w[:-1]) + 1
    bounds = np.r_[0, cuts, len(w)]
    return [(start + bounds[i], start + bounds[i + 1], int(w[bounds[i]])) for i in range(len(bounds) - 1)]


def nearest(m, f):
    """(team, distance m) of the slot nearest the ball at frame f, or (None, nan)."""
    if np.isnan(m["bx"][f]):
        return None, np.nan
    dx = (m["px"][f] - m["bx"][f]) * PITCH[0]
    dy = (m["py"][f] - m["by"][f]) * PITCH[1]
    d = np.hypot(dx, dy)
    if np.all(np.isnan(d)):
        return None, np.nan
    s = int(np.nanargmin(d))
    return int(m["slot_team"][s]), float(d[s])


def ball_speed(m, f):
    """Speed from finite differences of position, m/s; nan if either frame is missing."""
    if f < 1 or np.isnan(m["bx"][f]) or np.isnan(m["bx"][f - 1]):
        return np.nan
    return float(np.hypot((m["bx"][f] - m["bx"][f - 1]) * PITCH[0],
                          (m["by"][f] - m["by"][f - 1]) * PITCH[1]) * FPS)


def main():
    os.makedirs(OUT, exist_ok=True)
    m = load_tactical()
    gt = m["gt_public"]
    segs = segments(gt, *WINDOW)

    print("{} segments in the public window\n".format(len(segs)))
    print(" #  frame   from->to  len_before len_after | ball vis -3s..+3s | nearest team at t-1s / t / +1s "
          "(dist m) | ball in pitch | ball speed t")
    rows = []
    k = int(CONTEXT_S * FPS)
    for i in range(1, len(segs)):
        f = segs[i][0]
        a, b = LABELS[segs[i - 1][2]], LABELS[segs[i][2]]
        vis = ~np.isnan(m["bx"][f - k:f + k])
        near = [nearest(m, f + o) for o in (-FPS, 0, FPS)]
        inside = (0 <= m["bx"][f] <= 1 and 0 <= m["by"][f] <= 1) if not np.isnan(m["bx"][f]) else None
        rows.append((i, f, a, b, (segs[i - 1][1] - segs[i - 1][0]) / FPS, (segs[i][1] - segs[i][0]) / FPS,
                     vis.mean(), near, inside, ball_speed(m, f)))
        print("{:2d} {:6d}   {} -> {}   {:6.2f}s   {:6.2f}s  |  {:4.0%} ({}|{}) | {} | {} | {}".format(
            i, f, a, b, rows[-1][4], rows[-1][5], vis.mean(),
            "".join("#" if v else "." for v in vis[::FPS // 2][:6]),
            "".join("#" if v else "." for v in vis[::FPS // 2][6:]),
            "  ".join("{}({})".format("HA"[t] if t is not None else "-", "{:.1f}".format(d) if d == d else "--")
                      for t, d in near),
            {True: "in ", False: "OUT", None: " ? "}[inside],
            "{:.1f}".format(rows[-1][9]) if rows[-1][9] == rows[-1][9] else "--"))

    with open(os.path.join(OUT, "transitions.csv"), "w") as handle:
        handle.write("idx,frame,from,to,before_s,after_s,ball_vis_6s,near_m1_team,near_m1_d,near_0_team,"
                     "near_0_d,near_p1_team,near_p1_d,ball_inside,ball_speed\n")
        for i, f, a, b, before, after, vis, near, inside, speed in rows:
            flat = ",".join("{},{}".format("" if t is None else "HA"[t], "" if d != d else "{:.2f}".format(d))
                            for t, d in near)
            handle.write("{},{},{},{},{:.2f},{:.2f},{:.2f},{},{},{}\n".format(
                i, f, a, b, before, after, vis, flat, "" if inside is None else int(inside),
                "" if speed != speed else "{:.2f}".format(speed)))
    print("\nwrote", os.path.relpath(os.path.join(OUT, "transitions.csv"), ROOT))


if __name__ == "__main__":
    main()
