"""P9fk: the semi-Markov decoder's four main settings re-picked on SoccerTrack's full chain with P9fh + P9fi (the
production chain shape: video team evidence, video dead ball at 1/2, D weight 1.0). P9dh / P9dm picked them on DFL +
Metrica without video. Grid: emission scale, boundary gamma, turnover penalty, duration weight (3 x 3 x 3 x 3).

    .venv/bin/python analysis/p9fk_decoder_grid.py

Registered before the run (16 Nov): adopted only if the nested pick beats production (4, 6, 10, 2) by >= +0.003 at both
referees and on >= 6 of 8 matches.
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

GRID = list(itertools.product((3.0, 4.0, 5.0), (4.0, 6.0, 8.0), (6.0, 10.0, 14.0), (1.5, 2.0, 3.0)))
PROD = (4.0, 6.0, 10.0, 2.0)


def one(args):
    f, (es, bg, pt, ad) = args
    from mtah import semimarkov as sm
    from p9l_rescore_calibrated import merged
    z = np.load(f)
    g, lab = z["g"], z["lab"]
    d2, d3 = z["dead2"].astype(np.float64), z["dead3"].astype(np.float64)
    dead = 0.5 * np.clip(3 * d3 - 2 * d2, 0, 1) + 0.5 * d2
    cfg = copy.deepcopy(sm.CONFIG)
    d = cfg["decoder"]
    d["emission_weights"][2] = 1.0
    d["emission_scale"], d["boundary_gamma"], d["penalty_turnover"], d["alpha_duration"] = es, bg, pt, ad
    c = np.where(g, sm.decode_halves(z["team"].astype(np.float64), dead, g, z["b"].astype(np.float64), cfg=cfg), pm.X).astype(np.int8)
    return [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, merged(lab, 3.0))]


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "cache", "p9di", "in_*.npz")))
    with Pool(3) as pool:
        r = pool.map(one, [(f, c) for c in GRID for f in files], chunksize=4)
    S = np.array(r).reshape(len(GRID), len(files), 2).transpose(1, 0, 2)
    np.save(os.path.join(ROOT, "cache", "p9fk_scores.npy"), S)
    base = GRID.index(PROD)
    m = S.mean(0)
    for i in np.argsort(-m[:, 0])[:8]:
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
