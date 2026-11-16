"""P9fg: team members trained on recipe C plus SoccerTrack v2's ball-status labels.

Why (16 Nov, retrospective): SoccerTrack's labels are the labelled set closest to the organisers' public window in
possession statistics (per live minute 3.7-5.4 vs public 4.23; DFL at 1.6 s 6.7-8.5, Metrica 5.4-6.0; mean spell
11-16 s vs 14.2, DFL 7-9 s). P7z (v10 era) added SoccerTrack to the team GRU: SoccerTrack test 0.713 -> 0.811 (4/4)
and was dropped on the DFL judge (0.745 -> 0.730), the judge whose labelling is furthest from the organisers'.

Members: big, tcn128, set (single head), seeds 0-5, one draw, recipe C + SoccerTrack (mtah.members_c st=...).
  train <fold> <arch,...>   fold A = trained with SoccerTrack fold B's matches (scored on fold A), B = the reverse,
                            all = the eight labelled matches (production and the PFF judge)
  st                        SoccerTrack full chain as P9ev: v30's six members (12 seeds) vs the same plus the three
                            SoccerTrack members of the held-out fold's model (equal weight per member, 9 members)
  pff                       PFF World Cup as P9ew, the 'all' models

Registered before the runs (16 Nov): the three members go in as extra members only if SoccerTrack (cross-fold) is ahead
by >= +0.003 at both referees and on >= 5/8 matches at 1.6 s, and PFF is not behind by more than 0.005 at either
referee (PFF's labels are event-derived by our code, a different convention).
Ruling (16 Nov, before any result): Kaggle's weekly GPU quota is used up, training is local and memory-bound, so the
rule is applied first to one member (P9FG_ARCHES=tcn128, the fastest); the same thresholds.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.members_c import ST_LABELLED, cache_path, external_gru_c  # noqa: E402

ARCHES_ST = tuple(os.environ.get("P9FG_ARCHES", "big,tcn128,set").split(","))
SEEDS = tuple(range(6))


def fold_ids(fold):
    from p9cs_st_video_dead import TEST
    return {"A": TEST["B"], "B": TEST["A"], "all": ST_LABELLED}[fold]


def train(fold, arches):
    for a in arches.split(","):
        external_gru_c("cuda", arch=a, multitask=False, seeds=SEEDS, draws=1, st=fold_ids(fold))
        print("trained", a, fold, os.path.basename(cache_path("team", a, False, SEEDS, 1, st=fold_ids(fold))), flush=True)


def st_members(fold):
    G.train = __import__("p9bw_st_chain_base").no_training
    for a in ARCHES_ST:
        f = cache_path("team", a, False, SEEDS, 1, st=fold_ids(fold))
        assert os.path.exists(f), "missing " + os.path.basename(f)
    return [external_gru_c(arch=a, multitask=False, seeds=SEEDS, draws=1, st=fold_ids(fold)) for a in ARCHES_ST]


def both(ms, extra, m):
    """(v30's team, v30's six members + the SoccerTrack members, equal weight per member)."""
    pc = [np.mean([G.predict(e, m) for e in sets], axis=0) for sets in ms]
    pe = [G.predict(e, m) for e in extra]
    return np.mean(pc, axis=0), np.mean(pc + pe, axis=0)


def st():
    from mtah import semimarkov as sm
    from mtah.degrade import degrade, rates_for
    from mtah.external import from_soccertrack
    from p9cs_st_video_dead import TEST
    from p9ei_st_members_c import logit
    from p9ev_seeds12 import members
    from p9l_rescore_calibrated import merged
    ms = members()
    rows = {}
    for fold in ("A", "B"):
        extra = st_members(fold)
        for mid in TEST[fold]:
            assert mid not in fold_ids(fold)
            z = dict(np.load(os.path.join(ROOT, "cache", "p9di", "in_{}.npz".format(mid))))
            m0 = from_soccertrack(mid, merge_s=1.6)
            m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
            g, lab, b, dead = z["g"], z["lab"], z["b"].astype(np.float64), z["dead3"].astype(np.float64)
            e = logit(z["team"].astype(np.float64)) - logit(z["team0"].astype(np.float64))
            pc, px = both(ms, extra, m)
            l3 = merged(lab, 3.0)
            r = {}
            for k, p in (("v30", pc), ("+st", px)):
                codes = np.where(g, sm.decode_halves(1 / (1 + np.exp(-(logit(p) + e))), dead, g, b), pm.X).astype(np.int8)
                r[k] = [pm.spell_scores(L, codes, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
            rows[mid] = r
            print(mid, {k: np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    report("SoccerTrack", rows, 0.003)


def pff():
    from mtah.pff import from_pff, pff_ids
    from mtah.teammodel import decode
    from p9ev_seeds12 import members
    from p9l_rescore_calibrated import merged
    ms = members()
    extra = st_members("all")
    bnd = G.external_gru(target="boundary", swap=True, metrica=True)
    rows = {}
    for gid in pff_ids()[:12]:
        m = from_pff(gid, merge_s=1.6, ball="visible")
        g, lab = m["half"] > 0, m["labels"]
        l3 = merged(lab, 3.0)
        alive = g & np.isin(lab, (pm.H, pm.A))
        pc, px = both(ms, extra, m)
        sw = G.switch_costs(bnd, m)
        r = {}
        for k, p in (("v30", pc), ("+st", px)):
            c = np.where(g, np.where(alive, decode(p, alive, sw), pm.D), pm.X).astype(np.int8)
            r[k] = [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[gid] = r
        print(gid, {k: np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    report("PFF", rows, -0.005)


def report(name, rows, need):
    a, c = np.array([rows[k]["v30"] for k in rows]), np.array([rows[k]["+st"] for k in rows])
    d = (c - a).mean(0)
    n = int(((c - a)[:, 0] > 0).sum())
    ok = (d >= need).all() and (n >= 5 if need > 0 else True)
    print("{} {}: v30 {:.4f} / {:.4f}, +st {:.4f} / {:.4f}, diff {:+.4f} / {:+.4f} ({}/{} ahead at 1.6 s) -> {}".format(
        name, len(rows), *a.mean(0), *c.mean(0), *d, n, len(rows), "holds" if ok else "FAILS"))


if __name__ == "__main__":
    {"train": lambda: train(sys.argv[2], sys.argv[3]), "st": st, "pff": pff}[sys.argv[1]]()
