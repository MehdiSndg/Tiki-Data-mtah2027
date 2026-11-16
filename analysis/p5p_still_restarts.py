"""Inventory: "a still ball marks the restart" rested on one ball source and 7 goal kicks. Three
independent ball sources on our match, every Smart Tagging set piece type, and a control. GT-free.

    uv run python analysis/p5p_still_restarts.py      # -> eda_out/reports/P5p_still_restarts.md

Sources: the tactical ATD, the broadcast ATD (synchronised), ball_v1's strict-fp32 boxes (conf >=
0.5, not a painted spot). Still = mtah.timing.still_runs (< 0.5 m/s for >= 1 s). For every Smart
Tagging set piece outside the public window: a still run from 5 s before to 1 s after the tag's
start (P4d's window), per source and any source. Control: the same window around random alive
frames (raw-ATD classifier), away from any set piece by 40 s.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from mtah.broadcast import broadcast_ball_on_tactical  # noqa: E402
from mtah.context import SETPIECE_TYPES  # noqa: E402
from mtah.data import PITCH  # noqa: E402
from mtah.frames import load_frames  # noqa: E402
from mtah.match import from_tactical  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402
from mtah.timing import still_runs  # noqa: E402
from mtah.video_match import _hom, detector_ball  # noqa: E402
from p4_common import ours  # noqa: E402
from p4d_still_and_kinks import AFTER, BEFORE, runs_of  # noqa: E402

PUBLIC = (11185, 18685)
PASS = os.path.join(ROOT, "cache", "ball_v1_pass_blackwell")
OUT = os.path.join(ROOT, "eda_out", "reports", "P5p_still_restarts.md")
SEED, AWAY = 0, 40 * 25


def still(bx, by):
    s = np.zeros(len(bx), bool)
    for a, b in still_runs(bx, by, PITCH):
        s[a:b] = True
    return s


def main():
    m = from_tactical()
    n = len(m["bx"])
    df = load_frames()
    dead, _ = ours()
    xy, conf, spot = detector_ball(_hom(), n, PASS, 0, conf_min=0.5)
    dx, dy = np.where(spot, np.nan, xy[:, 0]), np.where(spot, np.nan, xy[:, 1])
    bbx, bby = broadcast_ball_on_tactical()
    S = {"taktik ATD": still(m["bx"], m["by"]), "yayın ATD": still(bbx, bby), "ball_v1": still(dx, dy)}
    S["herhangi biri"] = S["taktik ATD"] | S["yayın ATD"] | S["ball_v1"]
    kind = df["setpiece_type"].values
    sp = [(a, SETPIECE_TYPES[kind[a]]) for a, b in runs_of(df["setpiece"].values) if not (PUBLIC[0] <= a < PUBLIC[1])]
    near_sp = np.zeros(n, bool)
    for a, _ in sp:
        near_sp[max(0, a - AWAY):a + AWAY] = True
    alive = (m["half"] > 0) & ~np.isin(df["period"].values, NOT_PLAYED) & ~dead
    pool = np.flatnonzero(alive & ~near_sp)
    ctrl = np.random.RandomState(SEED).choice(pool, size=200, replace=False)
    names = list(S)
    lines = ["# P5p: duran toplarda hareketsiz top, üç top kaynağıyla (GT'siz)", "",
             "`analysis/p5p_still_restarts.py`. Pencere: etiket başından 5 s önce ile 1 s sonrası. Kontrol: duran toplardan "
             "40 s uzak, oyun içi 200 rastgele an.", "",
             "| tür | sayı | " + " | ".join(names) + " | iki kaynağın uyuştuğu |", "|---|---|" + "---|" * (len(names) + 1)]
    groups = [(t, [a for a, k in sp if k == t]) for t in ("goal_kick", "corner", "free_kick", "throw_in", "kickoff")]
    groups += [("hepsi", [a for a, _ in sp]), ("kontrol (oyun içi)", list(ctrl))]
    for t, starts in groups:
        if not starts:
            continue
        has = {k: np.array([S[k][max(0, a - BEFORE):a + AFTER].any() for a in starts]) for k in names}
        two = (has["taktik ATD"].astype(int) + has["yayın ATD"] + has["ball_v1"]) >= 2
        lines.append("| {} | {} | {} | {:.0%} |".format(t, len(starts), " | ".join("{:.0%}".format(has[k].mean()) for k in names), two.mean()))
        print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
