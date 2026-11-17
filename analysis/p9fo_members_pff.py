"""P9fo: a team member trained on recipe C plus PFF World Cup matches made to look like our ATD.

Why (16 Nov, data-use audit): PFF's own possession labels (event-derived, unmerged) are the labelled set closest to the
organisers' public window: per live minute 4.39 vs 4.23, mean spell 13.8 vs 14.2 s, median 7.6 vs 7.4 s, < 1.6 s 0.09 vs
0.12 (SoccerTrack 3.7-5.4, DFL at 1.6 s 6.7-8.5). Every earlier PFF training attempt (P9dc, P9dd, P9dl, P9ek, P9eq)
used PFF's raw broadcast ball, which drops 7-9 times a minute for ~0.8 s, unlike our ATD; here the ball is PFF's seen
or estimated one (fills the short dropouts) and the copy goes through our degradation (mtah.members_c pff=...).
Member: tcn128 single head, seeds 0-5, one draw; PFF games pff_ids()[12:22] (the judge's first 12 stay unseen).
  train          train the member
  st             SoccerTrack full chain as P9ev: v30's six members (12 seeds) vs the same plus this member (7 members)
  pff            PFF World Cup as P9ew on pff_ids()[:12] (held out), v30 vs + member

Registered before the runs (16 Nov): the member goes in as an extra member only if SoccerTrack is ahead by >= +0.003 at
both referees and on >= 5/8 matches at 1.6 s, and PFF (held out) is not behind at either referee.
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

ARCHES_ST = tuple(os.environ.get("P9FO_ARCH", "tcn128").split(","))     # P9fo2 (16 Nov, Colab): big / set with the same rule
SEEDS = tuple(range(6))


def pff_train():
    """P9fo: games [12:22]; P9fp (16 Nov, Colab, same rule): P9FO_PFF=12:64, every game but the judge's 12."""
    from mtah.pff import pff_ids
    a, b = map(int, os.environ.get("P9FO_PFF", "12:22").split(":"))
    return tuple(pff_ids()[a:b])


def train():
    for a in ARCHES_ST:
        external_gru_c("cuda", arch=a, multitask=False, seeds=SEEDS, draws=1, pff=pff_train())
        print("trained", os.path.basename(cache_path("team", a, False, SEEDS, 1, pff=pff_train())), flush=True)


def st_members(fold=None):
    G.train = __import__("p9bw_st_chain_base").no_training
    for a in ARCHES_ST:
        f = cache_path("team", a, False, SEEDS, 1, pff=pff_train())
        assert os.path.exists(f), "missing " + os.path.basename(f)
    return [external_gru_c(arch=a, multitask=False, seeds=SEEDS, draws=1, pff=pff_train()) for a in ARCHES_ST]


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
    extra = st_members()
    for fold in ("A", "B"):
        for mid in TEST[fold]:
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
    extra = st_members()
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
    report("PFF", rows, 0.0)


def report(name, rows, need):
    a, c = np.array([rows[k]["v30"] for k in rows]), np.array([rows[k]["+st"] for k in rows])
    d = (c - a).mean(0)
    n = int(((c - a)[:, 0] > 0).sum())
    ok = (d >= need).all() and (n >= 5 if need > 0 else True)
    print("{} {}: v30 {:.4f} / {:.4f}, +st {:.4f} / {:.4f}, diff {:+.4f} / {:+.4f} ({}/{} ahead at 1.6 s) -> {}".format(
        name, len(rows), *a.mean(0), *c.mean(0), *d, n, len(rows), "holds" if ok else "FAILS"))


if __name__ == "__main__":
    {"train": train, "st": st, "pff": pff}[sys.argv[1]]()
