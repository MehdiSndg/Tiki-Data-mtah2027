"""P9ev: judges for v30 (the six recipe-C members with twelve seeds each: 0-5 as v29 + 6-11 from P9eu) against v29
(six seeds), per P9eu's registered rule.
  st     SoccerTrack's full chain, as P9ei (evidence, dead, boundary and decoder fixed; only the team members change)
  video  our match, label-free: the members' call vs the seven video models' sign where the video is sure (P9ep)

    .venv/bin/python analysis/p9ev_seeds12.py st
    .venv/bin/python analysis/p9ev_seeds12.py video

Rule (P9eu, before the runs): v30 is adopted only if not behind v29 on SoccerTrack at either referee and its video
agreement (|e| > 1) is not lower. Inference only: a missing model stops the run.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.members_c import external_gru_c  # noqa: E402
from p9bw_st_chain_base import ARCHES, DRAWS, no_training  # noqa: E402

SETS = (tuple(range(6)), tuple(range(6, 12)))
OLD = int(os.environ.get("OLD_SETS", 1))          # P9ex: OLD_SETS=2 NEW_SETS=4 compares v30 (12 seeds) with v31 (24)
NEW = int(os.environ.get("NEW_SETS", 2))
SETS = tuple(tuple(range(a, a + 6)) for a in range(0, 6 * max(OLD, NEW), 6))


def members():
    G.train = no_training
    return [[external_gru_c(arch=a, multitask=mt, seeds=s, draws=DRAWS.get((a, mt), 1)) for s in SETS] for a in ARCHES for mt in (False, True)]


def team(ms, m):
    """(mean of the first OLD seed sets, mean of the first NEW seed sets) over the six members."""
    per = [[G.predict(e, m) for e in sets] for sets in ms]
    return np.mean([np.mean(p[:OLD], axis=0) for p in per], axis=0), np.mean([np.mean(p[:NEW], axis=0) for p in per], axis=0)


def st():
    from mtah import semimarkov as sm
    from mtah.degrade import degrade, rates_for
    from mtah.external import from_soccertrack
    from p9cs_st_video_dead import TEST
    from p9ei_st_members_c import logit
    from p9l_rescore_calibrated import merged
    ms = members()
    rows = {}
    for mid in [m for k in ("A", "B") for m in TEST[k]]:
        z = dict(np.load(os.path.join(ROOT, "cache", "p9di", "in_{}.npz".format(mid))))
        m0 = from_soccertrack(mid, merge_s=1.6)
        m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
        g, lab, b, dead = z["g"], z["lab"], z["b"].astype(np.float64), z["dead3"].astype(np.float64)
        e = logit(z["team"].astype(np.float64)) - logit(z["team0"].astype(np.float64))
        p6, p12 = team(ms, m)
        l3 = merged(lab, 3.0)
        r = {}
        for k, p in (("v29", p6), ("v30", p12)):
            codes = np.where(g, sm.decode_halves(1 / (1 + np.exp(-(logit(p) + e))), dead, g, b), pm.X).astype(np.int8)
            r[k] = [pm.spell_scores(L, codes, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[mid] = r
        print(mid, {k: np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    a, c = np.array([rows[m]["v29"] for m in rows]), np.array([rows[m]["v30"] for m in rows])
    d = (c - a).mean(0)
    print("SoccerTrack 8: v29 {:.4f} / {:.4f}, v30 {:.4f} / {:.4f}, diff {:+.4f} / {:+.4f} ({}/8 ahead at 1.6 s) -> {}".format(
        *a.mean(0), *c.mean(0), *d, int(((c - a)[:, 0] > 0).sum()), "holds" if (d >= 0).all() else "FAILS"))


def video():
    from mtah.match import from_tactical
    m = from_tactical()
    m["labels"] = np.zeros(len(m["half"]), np.int8)          # never read
    g = m["half"] > 0
    p6, p12 = team(members(), m)
    e = np.convolve(np.load(os.path.join(ROOT, "cache", "barath_dense_avg7.npz"))["e"].astype(np.float64), np.ones(151) / 151, "same")
    codes = np.load(os.path.join(ROOT, "cache", "v29_hsmm.npz"))["codes"]
    live = g & np.isin(codes, (pm.H, pm.A))
    for thr in (0.5, 1.0, 2.0):
        k = live & (np.abs(e) > thr)
        print("|e| > {}: v29 {:.4f} v30 {:.4f}".format(thr, *(np.mean((p[k] > 0.5) == (e[k] > 0)) for p in (p6, p12))))
    k = live & (np.abs(e) > 1.0)
    a, c = (np.mean((p[k] > 0.5) == (e[k] > 0)) for p in (p6, p12))
    print("rule (|e| > 1): v30 {:.4f} vs v29 {:.4f} -> {}".format(c, a, "holds" if c >= a else "FAILS"))


if __name__ == "__main__":
    {"st": st, "video": video}[sys.argv[1]]()
