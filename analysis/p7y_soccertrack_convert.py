"""Convert SoccerTrack v2 (Drive copy) to cached Match arrays and print each match's label statistics.

    MTAH_SOCCERTRACK=/content/drive/MyDrive/mtah2027/soccertrack python analysis/p7y_soccertrack_convert.py

Per match: graded minutes, possessions per 90, mean / median possession length, share under 1.6 s (what
DFL's merge_short removes), dead share, players seen per team, ball seen. Writes cache/soccertrack_<id>.npz.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.external import from_soccertrack, soccertrack_ids  # noqa: E402


def main():
    for mid in soccertrack_ids():
        m = from_soccertrack(mid)
        g = m["half"] > 0
        lab = m["labels"]
        sp = pm.possession_spells(lab, pm.valid_runs(lab))
        L = np.array([(b - a) / 25 for a, b, _ in sp])
        mins = g.sum() / 25 / 60
        vis = [np.nanmean((~np.isnan(m["px"][g][:, m["slot_team"] == t])).sum(1)) for t in (0, 1)]
        print("{} min {:.0f} poss/90 {:.0f} median {:.1f}s <1.6s {:.0%} D {:.1%} home {:.1%} players {:.1f}/{:.1f} ball {:.1%} sign {}".format(
            mid, mins, len(sp) / mins * 90, np.median(L), np.mean(L < 1.6), np.mean(lab[g] == pm.D),
            np.mean(lab[g] == pm.H) / max(np.mean(np.isin(lab[g], (pm.H, pm.A))), 1e-9), vis[0], vis[1],
            np.mean(~np.isnan(m["bx"][g])), m["sign"]), flush=True)


if __name__ == "__main__":
    main()
