"""P9ec: the friend's v9 "probabilities smoothed over +-1 s" before decoding, on v26's fused out-of-fold probabilities
(semi-Markov decoder), DFL 7 + Metrica 3, our labels at 1.6 s / 3 s. Box filter on the team log-odds over k frames
(k = 13, 25, 51), dead probability untouched.

    .venv/bin/python analysis/p9ec_prob_smooth.py

Registered before the run (13 Nov): k picked per match on the other nine; adopt only if nested DFL is ahead at 1.6 s and
neither DFL 3 s nor Metrica at either referee is behind.
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import semimarkov as sm  # noqa: E402
from p9de_team_fuse import C, lg, sg, theirs  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

KS = (13, 25, 51)


def box(x, k, g):
    out = x.copy()
    e = np.diff(np.r_[0, g.astype(np.int8), 0])
    for a, c in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        out[a:c] = np.convolve(np.pad(x[a:c], k // 2, mode="edge"), np.ones(k) / k, "valid")
    return out


def one(f):
    n = os.path.basename(f)
    z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f))
    g, lab, name = z["g"], z["labels"], n[4:-4]
    l3 = merged(lab, 3.0)
    b = np.nan_to_num(z["b"])
    p = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
    M = theirs("mehmet_ali_topkara", name, g)
    zt = 0.75 * lg(p) + 0.25 * lg(M["team"])
    dead = (np.nan_to_num(z["qh"]) + np.nan_to_num(z["qg"]) + M["p_dead"]) / 3
    sc = lambda team: [pm.spell_scores(L, np.where(g, sm.decode_halves(team, dead, g, b), pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
    r = {"base": sc(sg(zt))}
    for k in KS:
        r[k] = sc(sg(box(zt, k, g)))
    print(name, {kk: np.round(v, 4).tolist() for kk, v in r.items()}, flush=True)
    return name, r


def main():
    with Pool(4) as pool:
        res = dict(pool.map(one, sorted(glob.glob(C("oof_tcn128d3", "*.npz")))))
    names = sorted(res)
    base = np.array([res[m]["base"] for m in names])
    arms = {k: np.array([res[m][k] for m in names]) for k in KS}
    arms["nested"] = np.array([res[m][max(KS, key=lambda k: np.mean([np.mean(res[o][k]) - np.mean(res[o]["base"]) for o in names if o != m]))] for m in names])
    for k, a in arms.items():
        out = [str(k)]
        for src in ("DFL", "metrica"):
            ix = [i for i, m in enumerate(names) if src in m]
            d = a[ix] - base[ix]
            out.append("{} {:.4f} / {:.4f}  {:+.4f} / {:+.4f} ({}/{})".format(src, *a[ix].mean(0), *d.mean(0), int((d[:, 0] > 0).sum()), len(ix)))
        print(" | ".join(out))


if __name__ == "__main__":
    main()
