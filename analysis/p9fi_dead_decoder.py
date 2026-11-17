"""P9fi: the semi-Markov decoder's dead-ball settings on SoccerTrack's full chain, on top of P9fh (video dead weight 1/2).
Grid: emission weight of D (production 1.15), dead minimum duration in frames (60), dead median duration (350).
Leave-one-match-out pick.

    .venv/bin/python analysis/p9fi_dead_decoder.py

Registered before the run (16 Nov): a new setting is adopted only if the nested pick beats the production setting (with
P9fh) by >= +0.003 at both referees and on >= 6 of 8 matches.
"""
import copy
import glob
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402

GRID = list(itertools.product((1.0, 1.15, 1.3, 1.5), (25, 60), (250, 350, 500)))
PROD = (1.15, 60, 350)
if os.environ.get("P9FI2"):            # P9fi2 (registered 16 Nov after P9fi picked the grid's edge 1.0): lower D weights,
    GRID = [(w, 60, 350) for w in (0.7, 0.8, 0.9, 1.0)]    # durations as production; adopt below 1.0 only if the nested
    PROD = (1.0, 60, 350)                                  # pick beats 1.0 by >= +0.003 at both referees, >= 6/8


def one(args):
    f, (ew, dmin, dmed) = args
    from mtah import semimarkov as sm
    from p9l_rescore_calibrated import merged
    z = np.load(f)
    g, lab = z["g"], z["lab"]
    d2, d3 = z["dead2"].astype(np.float64), z["dead3"].astype(np.float64)
    dead = 0.5 * np.clip(3 * d3 - 2 * d2, 0, 1) + 0.5 * d2
    cfg = copy.deepcopy(sm.CONFIG)
    cfg["decoder"]["emission_weights"][2] = ew
    cfg["decoder"]["duration"]["dead"]["min"] = dmin
    cfg["decoder"]["duration"]["dead"]["median"] = dmed
    c = np.where(g, sm.decode_halves(z["team"].astype(np.float64), dead, g, z["b"].astype(np.float64), cfg=cfg), pm.X).astype(np.int8)
    return [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, merged(lab, 3.0))]


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "cache", "p9di", "in_*.npz")))
    with Pool(3) as pool:
        r = pool.map(one, [(f, c) for c in GRID for f in files])
    S = np.array(r).reshape(len(GRID), len(files), 2).transpose(1, 0, 2)      # match x config x referee
    base = GRID.index(PROD)
    m = S.mean(0)
    for i in np.argsort(-m[:, 0])[:6]:
        print(GRID[i], "{:.4f} / {:.4f}".format(*m[i]))
    print("production", PROD, "{:.4f} / {:.4f}".format(*m[base]))
    nested, picks = [], []
    for i in range(len(files)):
        keep = [j for j in range(len(files)) if j != i]
        k = int(np.argmax(S[keep].mean(axis=(0, 2))))
        picks.append(GRID[k])
        nested.append(S[i, k])
    d = np.array(nested) - S[:, base]
    ok = (d.mean(0) >= 0.003).all() and int((d[:, 0] > 0).sum()) >= 6
    print("fold picks", picks)
    print("nested diff {:+.4f} / {:+.4f} ({}/8 ahead) -> {}".format(*d.mean(0), int((d[:, 0] > 0).sum()), "holds" if ok else "FAILS"))


if __name__ == "__main__":
    main()
