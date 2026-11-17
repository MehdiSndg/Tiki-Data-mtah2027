"""P9fs: every candidate version scored end to end on the same judges (17 Nov; before, each change was measured against
v30 alone). Versions:
  v30  six recipe-C members (12 seeds), dead = (classifier + GRU + video) / 3, D weight 1.15
  v33  v30's members, dead = video / 2 + (classifier + GRU) / 4, D weight 1.0 (P9fh + P9fi)
  v37  v33 + P9fg's three SoccerTrack members (tcn128, big, set; on SoccerTrack the cross-fold models)
  v38  v37 + P9fo's PFF-trained set member
  v39  v33 + P9fo's PFF-trained set member
Members are averaged with equal weight (as make_submission). Inputs as P9ev / P9di (team evidence, boundary, dead from
cache/p9di). Reporting only: nothing is chosen here beyond the registered verdicts.

    .venv/bin/python analysis/p9fs_versions.py st
    .venv/bin/python analysis/p9fs_versions.py pff
"""
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.members_c import ST_LABELLED, external_gru_c  # noqa: E402

VERSIONS = ("v30", "v33", "v37", "v38", "v39")
SEEDS = tuple(range(6))


def extra(fold):
    """{'st': three SoccerTrack members (fold model or all), 'pff': the PFF set member}."""
    from p9bw_st_chain_base import no_training
    from mtah.pff import pff_ids
    G.train = no_training
    if fold == "all":
        ids = ST_LABELLED
    else:
        from p9cs_st_video_dead import TEST
        ids = {"A": TEST["B"], "B": TEST["A"]}[fold]
    st = [external_gru_c(arch=a, multitask=False, seeds=SEEDS, draws=1, st=ids) for a in ("tcn128", "big", "set")]
    pff = external_gru_c(arch="set", multitask=False, seeds=SEEDS, draws=1, pff=tuple(pff_ids()[12:22]))
    return st, pff


def teams(ms, ex, m):
    """{version: team probability} from v30's members (ms, P9ev's) and the extra members."""
    st, pff = ex
    base = [np.mean([G.predict(e, m) for e in sets], axis=0) for sets in ms]
    ps = [G.predict(e, m) for e in st]
    pp = [G.predict(pff, m)]
    return {"v30": np.mean(base, 0), "v33": np.mean(base, 0), "v37": np.mean(base + ps, 0),
            "v38": np.mean(base + ps + pp, 0), "v39": np.mean(base + pp, 0)}


def st():
    from mtah import semimarkov as sm
    from mtah.degrade import degrade, rates_for
    from mtah.external import from_soccertrack
    from p9cs_st_video_dead import TEST
    from p9ei_st_members_c import logit
    from p9ev_seeds12 import members
    from p9l_rescore_calibrated import merged
    ms = members()
    cfg33 = copy.deepcopy(sm.CONFIG)
    cfg33["decoder"]["emission_weights"][2] = 1.0
    rows = {}
    for fold in ("A", "B"):
        ex = extra(fold)
        for mid in TEST[fold]:
            z = dict(np.load(os.path.join(ROOT, "cache", "p9di", "in_{}.npz".format(mid))))
            m0 = from_soccertrack(mid, merge_s=1.6)
            m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
            g, lab, b = z["g"], z["lab"], z["b"].astype(np.float64)
            d2, d3 = z["dead2"].astype(np.float64), z["dead3"].astype(np.float64)
            dead33 = 0.5 * np.clip(3 * d3 - 2 * d2, 0, 1) + 0.5 * d2
            e = logit(z["team"].astype(np.float64)) - logit(z["team0"].astype(np.float64))
            l3 = merged(lab, 3.0)
            r = {}
            for v, p in teams(ms, ex, m).items():
                dead, cfg = (d3, sm.CONFIG) if v == "v30" else (dead33, cfg33)
                c = np.where(g, sm.decode_halves(1 / (1 + np.exp(-(logit(p) + e))), dead, g, b, cfg=cfg), pm.X).astype(np.int8)
                r[v] = [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
            rows[mid] = r
            print(mid, {v: np.round(x, 4).tolist() for v, x in r.items()}, flush=True)
    report("SoccerTrack", rows)


def pff():
    from mtah.pff import from_pff, pff_ids
    from mtah.teammodel import decode
    from p9ev_seeds12 import members
    from p9l_rescore_calibrated import merged
    ms = members()
    ex = extra("all")
    bnd = G.external_gru(target="boundary", swap=True, metrica=True)
    rows = {}
    for gid in pff_ids()[:12]:
        m = from_pff(gid, merge_s=1.6, ball="visible")
        g, lab = m["half"] > 0, m["labels"]
        l3 = merged(lab, 3.0)
        alive = g & np.isin(lab, (pm.H, pm.A))
        sw = G.switch_costs(bnd, m)
        r = {}
        for v, p in teams(ms, ex, m).items():
            if v == "v33":
                continue                                   # same team call as v30 (dead from the labels here)
            c = np.where(g, np.where(alive, decode(p, alive, sw), pm.D), pm.X).astype(np.int8)
            r[v] = [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[gid] = r
        print(gid, {v: np.round(x, 4).tolist() for v, x in r.items()}, flush=True)
    report("PFF (team call only; v33 = v30 here)", rows)


def report(name, rows):
    vs = list(next(iter(rows.values())).keys())
    S = np.array([[rows[k][v] for v in vs] for k in rows])
    print(name, len(rows), "matches")
    for i, v in enumerate(vs):
        d = (S[:, i] - S[:, 0]).mean(0)
        print("  {}  {:.4f} / {:.4f}   vs v30 {:+.4f} / {:+.4f}  ({}/{} ahead at 1.6 s)".format(
            v, *S[:, i].mean(0), *d, int((S[:, i, 0] > S[:, 0, 0]).sum()), len(rows)))


if __name__ == "__main__":
    {"st": st, "pff": pff}[sys.argv[1]]()
