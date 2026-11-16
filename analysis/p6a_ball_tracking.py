"""Step 1 of the modelling plan: does WASB-style online tracking clean ball_v1's track? GT-free.

    uv run python analysis/p6a_ball_tracking.py      # -> eda_out/reports/P6a_ball_tracking.md

ball_v1's boxes from the full strict-fp32 pass (cache/ball_v1_pass), on the pitch and off the
painted spots, projected to metres. Two tracks: the most confident box per frame (conf >= 0.3, what
the team call would have used) and mtah.balltrack.track_detections over every box. Measured only
on ball_v1's validation blocks (every 5th 10-minute block, public window out), against the ATD
ball where it is seen: kinks per seen minute (mtah.timing.kinks; the ATD's own 30-47), recall
(ATD ball seen and slow, track within 1.5 m), precision (track within 1.5 m where the ATD sees the
ball), and how much of the in-play ATD gaps (v1 chain says H/A) the track fills.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from mtah.balltrack import track_detections  # noqa: E402
from mtah.data import PITCH, load_tactical  # noqa: E402
from mtah.timing import kinks  # noqa: E402
from p3a_yolo_ball import on_pitch, on_spot, to_pitch  # noqa: E402
from p3e_ball_pseudolabels import BLOCK, PUBLIC  # noqa: E402

PASS = os.path.join(ROOT, "cache", "ball_v1_pass")
NEAR_M = 1.5
OUT = os.path.join(ROOT, "eda_out", "reports", "P6a_ball_tracking.md")


def boxes(m):
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(PASS, "chunk_*.npz")))]
    b = pd.DataFrame({c: np.concatenate([p[c] for p in parts]) for c in ("frame", "cls", "conf", "x", "y")})
    b = b[b.cls == 0].reset_index(drop=True)
    b = b[on_pitch(b, m["hom"]) & ~on_spot(b, m["hom"])].reset_index(drop=True)
    b["X"], b["Y"] = to_pitch(b, m["hom"])
    return b


def best_per_frame(b, n, conf=0.3):
    s = b[b.conf >= conf].sort_values("conf").groupby("frame").tail(1)
    X, Y = np.full(n, np.nan), np.full(n, np.nan)
    X[s.frame.values], Y[s.frame.values] = s.X.values, s.Y.values
    return X, Y


def measure(X, Y, m, val, alive_gap):
    ax, ay = m["bx"] * PITCH[0], m["by"] * PITCH[1]
    seen_atd = val[~np.isnan(ax[val])]
    slow = seen_atd[np.nan_to_num(m["bs"][seen_atd], nan=99) < 8]
    d = np.hypot(X - ax, Y - ay)
    have = ~np.isnan(X)
    out = {"recall": np.mean(d[slow] <= NEAR_M),
           "precision": np.mean(d[seen_atd][have[seen_atd]] <= NEAR_M),
           "gap_fill": np.mean(have[alive_gap]), "coverage": np.mean(have[val])}
    k, mins = 0, 0.0
    for a in np.unique(val // BLOCK):
        f = val[val // BLOCK == a]
        x, y = pd.Series(X[f]).interpolate(limit=5).values / PITCH[0], pd.Series(Y[f]).interpolate(limit=5).values / PITCH[1]
        kk = kinks(x, y, PITCH)
        k += np.sum(have[f][kk]) if len(kk) else 0
        mins += have[f].sum() / 25 / 60
    out["kinks_per_min"] = k / max(mins, 1e-9)
    return out


def main():
    m = load_tactical()
    n = len(m["bx"])
    g = np.flatnonzero(m["graded"])
    val = g[((g // BLOCK) % 5 == 4) & ((g < PUBLIC[0]) | (g > PUBLIC[1]))]
    v1 = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "v1_chain.csv")).label.values
    alive_gap = val[np.isnan(m["bx"][val]) & np.isin(v1[val], ["H", "A"])]
    b = boxes(m)
    print(len(b), "boxes", flush=True)
    atd = measure(m["bx"] * PITCH[0], m["by"] * PITCH[1], m, val, alive_gap)
    rows = [("ATD topu (referans)", atd)]
    for c in (0.3, 0.5):
        rows.append(("en güvenli kutu, conf >= {}".format(c), measure(*best_per_frame(b, n, c), m, val, alive_gap)))
    for cmin in (0.1, 0.2, 0.3):
        s = b[b.conf >= cmin]
        X, Y, _ = track_detections(s.frame.values, s.X.values, s.Y.values, s.conf.values, n)
        rows.append(("çevrimiçi takip, aday conf >= {}".format(cmin), measure(X, Y, m, val, alive_gap)))
        print(rows[-1], flush=True)
    lines = ["# P6a: ball_v1'e çevrimiçi takip (WASB usulü), GT'siz", "",
             "`analysis/p6a_ball_tracking.py`. ball_v1 doğrulama blokları ({} kare, public hariç); oyun içi ATD boşluğu {} kare.".format(len(val), len(alive_gap)), "",
             "| iz | görülen dakikada kırılma | recall (ATD yavaş, ≤ 1.5 m) | precision (≤ 1.5 m) | oyun içi boşluk doldurma | kapsama |",
             "|---|---|---|---|---|---|"]
    for name, r in rows:
        lines.append("| {} | {:.0f} | {:.1%} | {:.1%} | {:.1%} | {:.1%} |".format(
            name, r["kinks_per_min"], r["recall"], r["precision"], r["gap_fill"], r["coverage"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
