"""P9fc: the semi-Markov decoder's setting re-picked on SoccerTrack's full chain with the production probabilities.
P9dh / P9dm picked emission scale, boundary gamma, turnover penalty and duration weight on DFL + Metrica without the
video channel (v24 / v26 era); since then the probabilities changed (seven video models, recipe C, 12 seeds, Mali).
SoccerTrack (8 matches, cross-fold video, degraded as our match) is the only judge with the video channel.
  cache  v30's inputs per match (team members as P9ev with 12 seeds + P9di's evidence, dead, boundary) -> cache/p9fc/
  grid   every setting of the grid on every match; leave-one-match-out pick (mean of both referees on the other 7)

    .venv/bin/python analysis/p9fc_st_decoder.py cache
    .venv/bin/python analysis/p9fc_st_decoder.py grid

Registered before the run (16 Nov): a new setting is adopted only if the leave-one-match-out pick beats the production
setting by >= +0.003 at both referees and on >= 5 of 8 matches at 1.6 s, and the setting picked on all 8 is the same
one in at least 5 of the 8 folds (stability). Everything else stays. Labels of our match are never read.
"""
import copy
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(ROOT, "cache", "p9fc")
GRID = {"emission_scale": (3.0, 4.0, 5.0), "boundary_gamma": (4.0, 6.0, 8.0), "penalty_turnover": (6.0, 10.0, 14.0),
        "alpha_duration": (1.5, 2.0, 3.0)}
PROD = (4.0, 6.0, 10.0, 2.0)


def cache():
    from mtah.degrade import degrade, rates_for
    from mtah.external import from_soccertrack
    from p9cs_st_video_dead import TEST
    from p9ei_st_members_c import logit
    from p9ev_seeds12 import members, team
    from p9l_rescore_calibrated import merged
    os.makedirs(OUT, exist_ok=True)
    ms = members()
    for mid in [m for k in ("A", "B") for m in TEST[k]]:
        z = dict(np.load(os.path.join(ROOT, "cache", "p9di", "in_{}.npz".format(mid))))
        m0 = from_soccertrack(mid, merge_s=1.6)
        m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
        e = logit(z["team"].astype(np.float64)) - logit(z["team0"].astype(np.float64))
        p = 1 / (1 + np.exp(-(logit(team(ms, m)[1]) + e)))
        np.savez_compressed(os.path.join(OUT, mid + ".npz"), p=p, dead=z["dead3"], g=z["g"], b=z["b"], lab=z["lab"],
                            l3=merged(z["lab"], 3.0))
        print("cached", mid, flush=True)


def _score(args):
    f, cfgv = args
    from mtah import semimarkov as sm
    z = np.load(f)
    cfg = copy.deepcopy(sm.CONFIG)
    d = cfg["decoder"]
    d["emission_scale"], d["boundary_gamma"], d["penalty_turnover"], d["alpha_duration"] = cfgv
    g = z["g"]
    c = np.where(g, sm.decode_halves(z["p"], z["dead"].astype(np.float64), g, z["b"].astype(np.float64), cfg=cfg), pm.X).astype(np.int8)
    return [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (z["lab"], z["l3"])]


def grid():
    files = sorted(os.path.join(OUT, f) for f in os.listdir(OUT) if f.endswith(".npz"))
    cfgs = list(itertools.product(*GRID.values()))
    assert PROD in cfgs
    with Pool(int(os.environ.get("P9FC_PROCS", 4))) as pool:
        r = pool.map(_score, [(f, c) for c in cfgs for f in files], chunksize=4)
    S = np.array(r).reshape(len(cfgs), len(files), 2)            # config x match x referee
    prod = cfgs.index(PROD)
    picks, nested = [], []
    for i in range(len(files)):
        keep = [j for j in range(len(files)) if j != i]
        k = int(np.argmax(S[:, keep].mean(axis=(1, 2))))
        picks.append(cfgs[k])
        nested.append(S[k, i])
    nested = np.array(nested)
    d = nested - S[prod]
    best_all = cfgs[int(np.argmax(S.mean(axis=(1, 2))))]
    stable = sum(p == best_all for p in picks)
    print("production", PROD, "{:.4f} / {:.4f}".format(*S[prod].mean(0)))
    print("picked on all 8", best_all, "{:.4f} / {:.4f}".format(*S[cfgs.index(best_all)].mean(0)))
    print("fold picks", picks)
    ok = (d.mean(0) >= 0.003).all() and int((d[:, 0] > 0).sum()) >= 5 and stable >= 5
    print("nested {:.4f} / {:.4f}, diff {:+.4f} / {:+.4f} ({}/8 ahead), stable {}/8 -> {}".format(
        *nested.mean(0), *d.mean(0), int((d[:, 0] > 0).sum()), stable, "holds" if ok else "FAILS"))
    np.save(os.path.join(OUT, "grid_scores.npy"), S)


if __name__ == "__main__":
    {"cache": cache, "grid": grid}[sys.argv[1]]()
