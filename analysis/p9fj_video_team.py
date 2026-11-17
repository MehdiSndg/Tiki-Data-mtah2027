"""P9fj: the video team evidence's weight and smoothing re-picked on SoccerTrack's full chain with P9fh + P9fi
(video dead weight 1/2, D emission weight 1.0). Production: 0.5 x box(151 frames). Grid: weight 0.25-1.0 x window
75-301. P9di's inputs; the raw evidence from the cross-fold video models. Leave-one-match-out pick.

    .venv/bin/python analysis/p9fj_video_team.py

Registered before the run (16 Nov): adopted only if the nested pick beats (0.5, 151) by >= +0.003 at both referees and
on >= 6 of 8 matches.
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

GRID = list(itertools.product((0.25, 0.5, 0.75, 1.0), (75, 151, 301)))
PROD = (0.5, 151)


def prep():
    from mtah.external import from_soccertrack
    from p9ck_st_dense_fusion import evidence
    from p9cs_st_video_dead import TEST
    out = os.path.join(ROOT, "cache", "p9fj")
    os.makedirs(out, exist_ok=True)
    for fold, d in (("A", "cache/p9ci_A"), ("B", "cache/p9ci_B")):
        for mid in TEST[fold]:
            np.save(os.path.join(out, mid + ".npy"), evidence(os.path.join(ROOT, d), mid, from_soccertrack(mid, merge_s=1.6)["half"]))


def one(args):
    mid, (w, win) = args
    from mtah import semimarkov as sm
    from p9l_rescore_calibrated import merged
    z = np.load(os.path.join(ROOT, "cache", "p9di", "in_{}.npz".format(mid)))
    e = np.load(os.path.join(ROOT, "cache", "p9fj", mid + ".npy"))
    g, lab = z["g"], z["lab"]
    t0 = np.clip(z["team0"].astype(np.float64), 1e-4, 1 - 1e-4)
    p = 1 / (1 + np.exp(-(np.log(t0 / (1 - t0)) + w * np.convolve(e, np.ones(win) / win, "same"))))
    d2, d3 = z["dead2"].astype(np.float64), z["dead3"].astype(np.float64)
    dead = 0.5 * np.clip(3 * d3 - 2 * d2, 0, 1) + 0.5 * d2
    cfg = copy.deepcopy(sm.CONFIG)
    cfg["decoder"]["emission_weights"][2] = 1.0
    c = np.where(g, sm.decode_halves(p, dead, g, z["b"].astype(np.float64), cfg=cfg), pm.X).astype(np.int8)
    return [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, merged(lab, 3.0))]


def main():
    from p9cs_st_video_dead import TEST
    if not os.path.isdir(os.path.join(ROOT, "cache", "p9fj")):
        prep()
    mids = [m for k in ("A", "B") for m in TEST[k]]
    with Pool(3) as pool:
        r = pool.map(one, [(m, c) for c in GRID for m in mids])
    S = np.array(r).reshape(len(GRID), len(mids), 2).transpose(1, 0, 2)
    base = GRID.index(PROD)
    m = S.mean(0)
    for i in np.argsort(-m[:, 0])[:6]:
        print(GRID[i], "{:.4f} / {:.4f}".format(*m[i]))
    print("production", PROD, "{:.4f} / {:.4f}".format(*m[base]))
    nested, picks = [], []
    for i in range(len(mids)):
        keep = [j for j in range(len(mids)) if j != i]
        k = int(np.argmax(S[keep].mean(axis=(0, 2))))
        picks.append(GRID[k])
        nested.append(S[i, k])
    d = np.array(nested) - S[:, base]
    ok = (d.mean(0) >= 0.003).all() and int((d[:, 0] > 0).sum()) >= 6
    print("fold picks", picks)
    print("nested diff {:+.4f} / {:+.4f} ({}/8 ahead) -> {}".format(*d.mean(0), int((d[:, 0] > 0).sum()), "holds" if ok else "FAILS"))


if __name__ == "__main__":
    main()
