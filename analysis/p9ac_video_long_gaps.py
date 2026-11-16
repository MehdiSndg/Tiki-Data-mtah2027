"""How much of the long-gap in-play ball can video give us? (P9y: the true ball on the in-play frames of ATD gaps >= 10 s
is worth +0.015 on DFL; P9aa: the players alone place it only to ~10 m.) On our match, GT-free: ATD ball gaps by
length, the share v14 calls in play, and on those in-play frames the share where ball_v1 (tactical video, most
confident box >= 0.3 on the pitch, cache/ball_v1_pass), the broadcast ATD ball (mtah.broadcast.broadcast_ball_on_tactical) or
either has a ball. No labels are read (v14's own calls only). CPU.

    python analysis/p9ac_video_long_gaps.py      # -> eda_out/reports/P9ac_video_long_gaps.md
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from mtah.broadcast import broadcast_ball_on_tactical  # noqa: E402
from mtah.data import load_tactical  # noqa: E402
from p6a_ball_tracking import best_per_frame, boxes  # noqa: E402

FPS = 25
BINS = ((0, 2), (2, 10), (10, 30), (30, 1e9))
OUT = os.path.join(ROOT, "eda_out", "reports", "P9ac_video_long_gaps.md")


def main():
    m = load_tactical()
    n = len(m["bx"])
    played = m["half"] > 0
    alive = np.isin(pd.read_csv(os.path.join(ROOT, "evaluation", "results", "v14_chain.csv")).label.values[:n], ["H", "A"])
    vx, _ = best_per_frame(boxes(m), n, 0.3)
    bpres = np.isfinite(broadcast_ball_on_tactical()[0][:n])
    lost = np.isnan(m["bx"]) & played
    glen = np.zeros(n)
    e = np.diff(np.r_[0, lost.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        glen[a:b] = (b - a) / FPS
    lines = ["# P9ac: uzun top boşluklarında videonun payı (bizim maç, GT'siz)", "", "`analysis/p9ac_video_long_gaps.py`.", "",
             "Boşluk uzunluğu: kayıp kare, v14 oyunda payı; oyundaki karelerde ball_v1 / yayın ATD / ikisinden biri", ""]
    for lo, hi in BINS:
        sel = lost & (glen >= lo) & (glen < hi)
        a = sel & alive
        v, bc = np.isfinite(vx) & a, bpres & a
        lines.append("- {}–{} s: {} kare, oyunda %{:.0f} ({} kare = {:.1f} dk); ball_v1 %{:.0f}, yayın %{:.0f}, biri %{:.0f}".format(
            lo, hi if hi < 1e9 else "∞", int(sel.sum()), 100 * a.sum() / max(1, sel.sum()), int(a.sum()), a.sum() / FPS / 60,
            100 * v.sum() / max(1, a.sum()), 100 * bc.sum() / max(1, a.sum()), 100 * (v | bc).sum() / max(1, a.sum())))
        print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
