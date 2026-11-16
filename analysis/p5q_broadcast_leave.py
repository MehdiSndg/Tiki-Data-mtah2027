"""Inventory: "the broadcast leaving the pitch marks the dead ball" (76% of D, 85% precision) came from
the public window only. A GT-free second look over the whole match.

    uv run python analysis/p5q_broadcast_leave.py      # -> eda_out/reports/P5q_broadcast_leave.md

Leaving = no valid broadcast homography (close-up, replay, crowd) on a tactical frame that has a
broadcast frame (cache/broadcast_on_tactical.npz). Shares of leaving frames: around Smart Tagging
set pieces (5 s before to 1 s after the tag's start; outside the public window), inside the raw-ATD
classifier's dead ball and alive play, and in alive play far (> 40 s) from any set piece.
Second part: the classifier's dead frames inside long tactical-ATD gaps (>= 10 s) where the
broadcast stays on the pitch and the broadcast ATD sees a ball moving faster than 2 m/s -- likely
play the classifier calls dead (P5n: its errors sit in long gaps).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from mtah.frames import load_frames  # noqa: E402
from mtah.periods import NOT_PLAYED  # noqa: E402
from p4_common import ours  # noqa: E402
from p4d_still_and_kinks import AFTER, BEFORE, runs_of  # noqa: E402

PUBLIC = (11185, 18685)
AWAY = 40 * 25
OUT = os.path.join(ROOT, "eda_out", "reports", "P5q_broadcast_leave.md")


def main():
    b = np.load(os.path.join(ROOT, "cache", "broadcast_on_tactical.npz"))
    have, leave = b["b_present"], b["b_present"] & ~b["b_valid"]
    df = load_frames()
    dead, _ = ours()
    played = (df["half"].values > 0) & ~np.isin(df["period"].values, NOT_PLAYED)
    outside = np.ones(len(dead), bool)
    outside[PUBLIC[0]:PUBLIC[1]] = False
    starts = [a for a, _ in runs_of(df["setpiece"].values) if outside[a]]
    near = np.zeros(len(dead), bool)
    win = np.zeros(len(dead), bool)
    for a in starts:
        near[max(0, a - AWAY):a + AWAY] = True
        win[max(0, a - BEFORE):a + AFTER] = True
    rows = [("duran top penceresi", win & played & have),
            ("sınıflandırıcının ölü topu", dead & played & have & outside),
            ("oyun içi (sınıflandırıcı)", ~dead & played & have & outside),
            ("oyun içi, duran toplardan > 40 s uzak", ~dead & played & have & outside & ~near)]
    lines = ["# P5q: yayının sahadan ayrılması, maçın tamamında (GT'siz)", "",
             "`analysis/p5q_broadcast_leave.py`. Ayrılma = yayın karesinde geçerli homografi yok.", "",
             "| kareler | kare | yayın sahadan ayrılmış |", "|---|---|---|"]
    for name, mask in rows:
        lines.append("| {} | {} | {:.0%} |".format(name, mask.sum(), leave[mask].mean()))
    pw = [leave[max(0, a - BEFORE):a + AFTER].any() for a in starts]
    lines += ["", "Duran topların {:.0%}'inde pencerede en az bir ayrılma karesi var ({} duran top).".format(np.mean(pw), len(starts)),
              "Ayrılma karelerinin {:.0%}'i sınıflandırıcının ölü topunda.".format((leave & dead & played & outside).sum() / max(1, (leave & played & outside).sum()))]
    from mtah.broadcast import broadcast_ball_on_tactical
    from mtah.data import PITCH
    from mtah.match import from_tactical
    m = from_tactical()
    bbx, bby = broadcast_ball_on_tactical()
    sp = np.r_[np.nan, np.hypot(np.diff(bbx) * PITCH[0], np.diff(bby) * PITCH[1]) * 25]
    sp = np.convolve(np.nan_to_num(sp, nan=0), np.ones(13) / 13, "same")
    miss = np.isnan(m["bx"]) & played & outside
    long_gap = np.zeros(len(miss), bool)
    for a, z in runs_of(miss):
        if z - a >= 10 * 25:
            long_gap[a:z] = True
    sel = dead & long_gap
    live_moving = sel & have & ~leave & ~np.isnan(bbx) & (sp > 2)
    lines += ["", "Taktik ATD'nin ≥ 10 s'lik boşluklarında sınıflandırıcının ölü dediği {} kare: {:.0%}'inde yayın sahada ve "
              "yayın ATD'si 2 m/s'den hızlı bir top görüyor ({} kare, oyunun muhtemelen sürdüğü anlar); yayın sahada ve "
              "top yok ya da yavaş {:.0%}; yayın ayrılmış {:.0%}.".format(
                  sel.sum(), live_moving.sum() / max(1, sel.sum()), live_moving.sum(),
                  (sel & have & ~leave & ~live_moving).sum() / max(1, sel.sum()), (sel & leave).sum() / max(1, sel.sum()))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
