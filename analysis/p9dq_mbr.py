"""P9dq: expected-spellF1 (minimum Bayes risk) decision over paths sampled from the semi-Markov model, on v26's fused
out-of-fold probabilities (our six members + Mali's team at 0.25, our two dead models + Mali's dead), DFL 7 + Metrica 3.
The friend's v22 idea; Mali's implementation (mtah/mbr.py, vendored unchanged) which he found neutral with his own
probabilities. Paths are sampled from the HSMM (explicit durations), unlike P9cr which sampled a frame-Markov chain.

    /home/mehdi/venvs/mali/bin/python analysis/p9dq_mbr.py      (numba needed)

Registered before the run (13 Nov): temperature grid {1, 2, 4, 8, 16}, 300 samples, seed 2027; temperature picked
per match on the other nine matches (mean of both referees); adopt only if nested DFL is ahead at 1.6 s, and neither
DFL at 3 s nor Metrica at either referee is behind.
"""
import glob
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import mbr, semimarkov as sm  # noqa: E402
from p9de_team_fuse import C, lg, sg, theirs  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

TEMPS = (1, 2, 4, 8, 16)
NS = int(os.environ.get("NS", 300))


def decode_run_mbr(lp, cfg, boundary, temperature):
    stride, dur, edge, dmin, trans = sm._tables(cfg)
    T = len(lp)
    S = int(math.ceil(T / stride))
    pad = S * stride - T
    lp_pad = np.vstack([lp, np.repeat(lp[-1:], pad, axis=0)]) if pad else lp
    step_lp = lp_pad.reshape(S, stride, 3).sum(axis=1)
    bonus = sm.boundary_bonus(boundary, stride, S, float(cfg["decoder"].get("boundary_gamma", 0.0)))
    adj = np.zeros((S + 1, 3))
    cstep, dur_c = sm.contested_duration(None, stride, S, cfg, dur.shape[1] - 1)
    vit = sm._viterbi(step_lp, dur, edge, dmin, trans, bonus, adj, cstep, dur_c)
    step = mbr.decode(step_lp, dur, edge, dmin, trans, bonus, adj, cstep, dur_c,
                      dict(samples=NS, temperature=temperature, seed=2027), viterbi_labels=vit)
    labels = np.repeat(step, stride)[:T].astype(np.int8)
    return sm._refine(labels, lp, int(cfg["decoder"]["refine_frames"]))


def decode_halves_mbr(p_home, p_dead, g, boundary, temperature, cfg=sm.CONFIG):
    P = np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], 1)
    out = np.full(len(g), 3, np.int8)
    e = np.diff(np.r_[0, np.asarray(g).astype(np.int8), 0])
    for a, c in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        out[a:c] = decode_run_mbr(sm.emission_logprob(P[a:c], cfg), cfg, np.asarray(boundary[a:c], np.float64), temperature)
    return out


def one(f):
    n = os.path.basename(f)
    z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f))
    g, lab, name = z["g"], z["labels"], n[4:-4]
    l3 = merged(lab, 3.0)
    b = np.nan_to_num(z["b"])
    p = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
    qh, qg = np.nan_to_num(z["qh"]), np.nan_to_num(z["qg"])
    M = theirs("mehmet_ali_topkara", name, g)
    team, dead = sg(0.75 * lg(p) + 0.25 * lg(M["team"])), (qh + qg + M["p_dead"]) / 3
    sc = lambda c: [pm.spell_scores(L, np.where(g, c, pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
    r = {"viterbi": sc(sm.decode_halves(team, dead, g, b))}
    for t in TEMPS:
        r[t] = sc(decode_halves_mbr(team, dead, g, b, t))
    print(name, {k: np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    return name, r


def main():
    files = sorted(glob.glob(C("oof_tcn128d3", "*.npz")))
    with Pool(int(os.environ.get("NP", 5))) as pool:
        res = dict(pool.map(one, files))
    names = sorted(res)
    base = np.array([res[m]["viterbi"] for m in names])
    nested = []
    for i, m in enumerate(names):
        others = [o for o in names if o != m]
        best = max(TEMPS, key=lambda t: np.mean([np.mean(res[o][t]) - np.mean(res[o]["viterbi"]) for o in others]))
        nested.append(res[m][best])
    arms = {t: np.array([res[m][t] for m in names]) for t in TEMPS}
    arms["nested"] = np.array(nested)
    for k, a in arms.items():
        out = [str(k)]
        for src in ("DFL", "metrica"):
            ix = [i for i, m in enumerate(names) if src in m]
            d = a[ix] - base[ix]
            out.append("{} {:.4f} / {:.4f}  {:+.4f} / {:+.4f} ({}/{})".format(src, *a[ix].mean(0), *d.mean(0), int((d[:, 0] > 0).sum()), len(ix)))
        print(" | ".join(out))


if __name__ == "__main__":
    main()
