"""How many possessions does our match get for each switch cost? (Our match under-splits: v14 ~250 per 90 min, while
the baseline-calibrated DFL labels have ~286 and 1.6 s-merged ones ~332, P9k / P9l.) The final team probability, the
production switch costs and the dead ball saved by make_submission (MTAH_DUMP) are re-decoded with the cost scaled by
each factor; the possession count per 90 min of play is printed. No ground truth is read. CPU, light.

    python analysis/p9s_density_scan.py <dump.npz> [...]
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.teammodel import decode  # noqa: E402

SCALES = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4)


def per90(codes, g):
    played = int(np.sum(g))
    return len(pm.possession_spells(codes, pm.valid_runs(np.where(g, codes, pm.X).astype(np.int8)))) * 90 * 60 * 25 / max(1, played)


def main(*paths):
    for path in paths:
        z = np.load(path)
        p, sw, dead, g = z["p"], z["switch"], z["dead"].astype(bool), z["g"].astype(bool)
        row = []
        for s in SCALES:
            codes = np.where(g, decode(p, g & ~dead, sw * s), pm.X).astype(np.int8)
            row.append("x{:g}: {:.0f}".format(s, per90(codes, g)))
        print(os.path.basename(path), "|", ", ".join(row), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
