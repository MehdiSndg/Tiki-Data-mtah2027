"""P9fb judges: the candidate's six recipe-C members (CAND_SETS seed sets of six: 2 = v30, 4 = v31) against the same
six plus four extra recipe-C members of the two other architectures (gru, tcn; single and two-head; seeds 0-5;
trained on Kaggle by analysis/p9eu_members_c_seeds.py), probability mean over all members, equal weight.
  st   SoccerTrack full chain as P9ev (evidence, dead, boundary and decoder fixed; only the team members change)
  pff  PFF World Cup as P9ew (team call only, dead from the labels, production boundary GRU)

    CAND_SETS=2 .venv/bin/python analysis/p9fb_extra_members.py st | pff

Rule (registered 15 Nov in p9eu's docstring, before the run): the extra members are adopted only if not behind the
candidate on SoccerTrack and on PFF at either referee. Inference only: a missing model stops the run.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.members_c import cache_path, external_gru_c  # noqa: E402
from p9bw_st_chain_base import ARCHES, DRAWS, no_training  # noqa: E402

CAND = int(os.environ.get("CAND_SETS", 2))
SETS = tuple(tuple(range(a, a + 6)) for a in range(0, 6 * CAND, 6))
EXTRA = [(a, mt) for a in ("gru", "tcn") for mt in (False, True)]


def members():
    G.train = no_training
    for a, mt in EXTRA:
        f = cache_path("team", a, mt, tuple(range(6)), 1)
        assert os.path.exists(f), "missing " + os.path.basename(f)
    cand = [[external_gru_c(arch=a, multitask=mt, seeds=s, draws=DRAWS.get((a, mt), 1)) for s in SETS] for a in ARCHES for mt in (False, True)]
    extra = [external_gru_c(arch=a, multitask=mt, seeds=tuple(range(6)), draws=1) for a, mt in EXTRA]
    return cand, extra


def team(ms, m):
    """(candidate: mean of its six members, each the mean of its seed sets; with extra: mean of all ten members)."""
    cand, extra = ms
    pc = [np.mean([G.predict(e, m) for e in sets], axis=0) for sets in cand]
    pe = [G.predict(e, m) for e in extra]
    return np.mean(pc, axis=0), np.mean(pc + pe, axis=0)


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
        pc, px = team(ms, m)
        l3 = merged(lab, 3.0)
        r = {}
        for k, p in (("cand", pc), ("extra", px)):
            codes = np.where(g, sm.decode_halves(1 / (1 + np.exp(-(logit(p) + e))), dead, g, b), pm.X).astype(np.int8)
            r[k] = [pm.spell_scores(L, codes, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[mid] = r
        print(mid, {k: np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    report("SoccerTrack", rows)


def pff():
    from mtah.pff import from_pff, pff_ids
    from mtah.teammodel import decode
    from p9l_rescore_calibrated import merged
    ms = members()
    bnd = G.external_gru(target="boundary", swap=True, metrica=True)
    rows = {}
    for gid in pff_ids()[:12]:
        m = from_pff(gid, merge_s=1.6, ball="visible")
        g, lab = m["half"] > 0, m["labels"]
        l3 = merged(lab, 3.0)
        alive = g & np.isin(lab, (pm.H, pm.A))
        pc, px = team(ms, m)
        sw = G.switch_costs(bnd, m)
        r = {}
        for k, p in (("cand", pc), ("extra", px)):
            c = np.where(g, np.where(alive, decode(p, alive, sw), pm.D), pm.X).astype(np.int8)
            r[k] = [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[gid] = r
        print(gid, {k: np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    report("PFF", rows)


def report(name, rows):
    a, c = np.array([rows[k]["cand"] for k in rows]), np.array([rows[k]["extra"] for k in rows])
    d = (c - a).mean(0)
    print("{} {} (candidate {} seed sets): cand {:.4f} / {:.4f}, +extra {:.4f} / {:.4f}, diff {:+.4f} / {:+.4f} ({}/{} ahead at 1.6 s) -> {}".format(
        name, len(rows), CAND, *a.mean(0), *c.mean(0), *d, int(((c - a)[:, 0] > 0).sum()), len(rows), "holds" if (d >= 0).all() else "FAILS"))


if __name__ == "__main__":
    {"st": st, "pff": pff}[sys.argv[1]]()
