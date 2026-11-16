"""Label / tracking statistics of PFF games (aggregates only): tools/pff_stats.py <id> [...]"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtah.io import pff  # noqa: E402


def runs(m):
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


for g in sys.argv[1:] or pff.games():
    t0 = time.time()
    d = pff.cached_game(g)
    lab, per = d.labels, d.period > 0
    played = per & (lab != 3)
    a, b = runs(per & (lab == 2))
    spl = np.concatenate([(lambda r: (r[1] - r[0]) / 25)(runs(per & (lab == k))) for k in (0, 1)])
    hid = np.isnan(d.ball[:, 0])
    print(g, f"t {time.time() - t0:.0f}s frames {per.sum()} pitch {d.pitch} N {d.players.shape[1]}",
          f"X {(per & (lab == 3)).sum() / per.sum():.3f} | D share {(lab[played] == 2).mean():.3f}",
          f"dead segs {len(a)} median {np.median((b - a) / 25):.1f}s | spells {len(spl)} median {np.median(spl):.1f}s",
          f"home {(lab[played] == 0).sum() / (lab[played] < 2).sum():.3f}",
          f"| ball hidden live {hid[played & (lab < 2)].mean():.2f} dead {hid[played & (lab == 2)].mean():.2f}",
          f"| players/frame {(~np.isnan(d.players[per][:, :, 0])).sum(1).mean():.1f}", flush=True)
