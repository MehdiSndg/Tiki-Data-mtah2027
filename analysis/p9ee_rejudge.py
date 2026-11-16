"""P9ee: every chain-level choice made on the production copies (whose ball reappears exactly at the restart), re-judged
on the realistic-dead copies (P9dt / P9ea). Per match (DFL 7 + Metrica 3): v26's fused team probability and boundary
(out of fold, unchanged), dead ball = mean of P9ea's held-out classifier and GRU probabilities on the judge's copy and
Mali's P(dead). Judges: "old" (production copy, prod classifier = v26) and "v6" (realistic copy, mix classifier = v27a).
Re-tests, each against its judge's base (v26's decoder and fusion):
  dec   the semi-Markov decoder grid (emission scale, boundary gamma, turnover penalty, duration weight, dead-ball median
        duration), nested pick
  mali  Mali's team weight 0 / 0.25 / 0.5 and his P(dead) in / out
  mbr   expected-spellF1 decision (P9dq), temperature 2 / 4 / 8, nested
  smo   team log-odds box filter 13 / 25 frames (P9ec), nested
Needs cache/p9ea (P9ea's dumps) and numba (Mali's venv).

    /home/mehdi/venvs/mali/bin/python analysis/p9ee_rejudge.py [dec|mali|mbr|smo ...]

Registered before the run (13 Nov), two independent judges (the user's rule: never one judge): on the v6 judge nested
DFL ahead at 1.6 s and neither DFL 3 s nor Metrica at either referee behind, AND on SoccerTrack's full chain (P9di's
inputs: real tracking, video channels, v26's dead ball; "st" below, 8 matches, same nested pick) not behind at either
referee. The old judge is reported only. "mali" has no SoccerTrack judge (his model never ran there).
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
from mtah import semimarkov as sm  # noqa: E402
from p9de_team_fuse import C, lg, sg, theirs  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402


def cfg_with(scale=4.0, gamma=6.0, turn=10.0, alpha=2.0, dmed=350):
    c = copy.deepcopy(sm.CONFIG)
    d = c["decoder"]
    d["emission_scale"], d["boundary_gamma"], d["penalty_turnover"], d["alpha_duration"] = scale, gamma, turn, alpha
    d["duration"]["dead"]["median"] = dmed
    return c


DEC = [dict(scale=s, gamma=gm, turn=t, alpha=a, dmed=dm) for s, gm, t, a, dm in itertools.product((3.0, 4.0, 5.0), (4.0, 6.0, 8.0), (6.0, 10.0, 14.0), (1.0, 2.0, 3.0), (250, 350, 500))]


def load(n):
    z, big, t1, d3 = (dict(np.load(C(x, n), allow_pickle=True)) for x in ("oof_v16", "oof_big", "oof_tcn128", "oof_tcn128d3"))
    e = dict(np.load(C("p9ea", n)))
    g, lab, name = z["g"], z["labels"], n[4:-4]
    p = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
    M = theirs("mehmet_ali_topkara", name, g)
    return dict(g=g, lab=lab, l3=merged(lab, 3.0), b=np.nan_to_num(z["b"]), p=p, mt=M["team"], md=M["p_dead"], e=e, name=name)


def dead_of(d, judge, mali=True):
    arm = "prod" if judge == "old" else "mix"
    parts = [d["e"]["p_{}_{}".format(judge, arm)], d["e"]["q_" + judge]] + ([d["md"]] if mali else [])
    return np.mean(parts, axis=0).astype(np.float64)


def team_of(d, w=0.25):
    return sg((1 - w) * lg(d["p"]) + w * lg(d["mt"]))


def score(d, codes):
    return [pm.spell_scores(L, np.where(d["g"], codes, pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (d["lab"], d["l3"])]


def arms_for(what, d, judge):
    g, b = d["g"], d["b"]
    dead, team = dead_of(d, judge), team_of(d)
    out = {"base": score(d, sm.decode_halves(team, dead, g, b))}
    if what == "dec":
        for i, kw in enumerate(DEC):
            out[i] = score(d, sm.decode_halves(team, dead, g, b, cfg=cfg_with(**kw)))
    elif what == "mali":
        for w in (0.0, 0.5):
            out["team {}".format(w)] = score(d, sm.decode_halves(team_of(d, w), dead, g, b))
        out["no mali dead"] = score(d, sm.decode_halves(team, dead_of(d, judge, False), g, b))
    elif what == "mbr":
        from p9dq_mbr import decode_halves_mbr
        for t in (2, 4, 8):
            out[t] = score(d, decode_halves_mbr(team, dead, g, b, t))
    elif what == "smo":
        from p9ec_prob_smooth import box
        for k in (13, 25):
            out[k] = score(d, sm.decode_halves(sg(box(lg(team), k, g)), dead, g, b))
    return out


def load_st(f):
    z = dict(np.load(f))
    g, lab = z["g"], z["lab"]
    return dict(g=g, lab=lab, l3=merged(lab, 3.0), b=z["b"].astype(np.float64), team=z["team"].astype(np.float64),
                dead=z["dead3"].astype(np.float64), name="ST" + os.path.basename(f)[3:-4])


def arms_st(what, d):
    g, b, team, dead = d["g"], d["b"], d["team"], d["dead"]
    out = {"base": score(d, sm.decode_halves(team, dead, g, b))}
    if what == "dec":
        for i, kw in enumerate(DEC):
            out[i] = score(d, sm.decode_halves(team, dead, g, b, cfg=cfg_with(**kw)))
    elif what == "mbr":
        from p9dq_mbr import decode_halves_mbr
        for t in (2, 4, 8):
            out[t] = score(d, decode_halves_mbr(team, dead, g, b, t))
    elif what == "smo":
        from p9ec_prob_smooth import box
        for k in (13, 25):
            out[k] = score(d, sm.decode_halves(sg(box(lg(team), k, g)), dead, g, b))
    return out


def one(args):
    what, f = args
    if os.path.basename(f).startswith("in_"):
        d = load_st(f)
        return d["name"], {"st": arms_st(what, d)}
    d = load(os.path.basename(f))
    return d["name"], {j: arms_for(what, d, j) for j in ("old", "v6")}


def report(what, res):
    print("==", what)
    for judge in ("old", "v6", "st"):
        names = sorted(m for m in res if judge in res[m])
        if not names:
            continue
        R = {m: res[m][judge] for m in names}
        base = np.array([R[m]["base"] for m in names])
        keys = [k for k in R[names[0]] if k != "base"]
        nested = np.array([R[m][max(keys, key=lambda k: np.mean([np.mean(R[o][k]) - np.mean(R[o]["base"]) for o in names if o != m]))] for m in names])
        rows = [("nested", nested)] + ([(k, np.array([R[m][k] for m in names])) for k in keys] if what != "dec" else [])
        if what == "dec":
            best = sorted(keys, key=lambda k: -np.mean([np.mean(R[o][k]) - np.mean(R[o]["base"]) for o in names]))[:5]
            rows += [("{} {}".format(k, DEC[k]), np.array([R[m][k] for m in names])) for k in best]
        srcs = ("ST",) if judge == "st" else ("DFL", "metrica")
        print("judge", judge, " ".join("{} base {:.4f} / {:.4f}".format(src, *base[[i for i, m in enumerate(names) if src in m]].mean(0)) for src in srcs))
        for k, a in rows:
            out = ["  " + str(k)]
            for src in srcs:
                ix = [i for i, m in enumerate(names) if src in m]
                dd = a[ix] - base[ix]
                out.append("{} {:+.4f} / {:+.4f} ({}/{})".format(src, *dd.mean(0), int((dd[:, 0] > 0).sum()), len(ix)))
            print(" | ".join(out), flush=True)


def main(*whats):
    files = sorted(glob.glob(C("p9ea", "oof_*.npz")))
    st = sorted(glob.glob(C("p9di", "in_*.npz")))
    for what in whats or ("mali", "smo", "mbr", "dec"):
        with Pool(int(os.environ.get("NP", 5))) as pool:
            res = dict(pool.map(one, [(what, f) for f in files + (st if what != "mali" else [])]))
        report(what, res)


if __name__ == "__main__":
    main(*sys.argv[1:])
