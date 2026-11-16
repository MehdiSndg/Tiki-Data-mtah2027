"""P9di: the HSMM decoder of P9dh (Mali's, configuration 4 / 6 / 10 / 2 fixed on DFL + Metrica) on SoccerTrack's full
chain with the video channels, as the submission uses them: team = v19c's chain + 0.5 x the 151-frame-smoothed dense
evidence, P(dead) = mean of v8's classifier, GRU and the video head (P9cs's arm "+ dense + dead"), boundary from the
chain's switch cost. Against our decoder on the same inputs. Cross-fold video predictions, 8 matches.

    python analysis/p9di_st_hsmm.py prep <fold A predictions> <fold B predictions>     # -> cache/p9di/in_<match>.npz
    <numba venv>/bin/python analysis/p9di_st_hsmm.py decode <Mali's code dir>          # -> cache/p9di/out_<match>.npz
    python analysis/p9di_st_hsmm.py score
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "cache", "p9di")
CFG = dict(emission_scale=4.0, boundary_gamma=6.0, penalty_turnover=10.0, alpha_duration=2.0)


def prep(dA, dB):
    sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
    import p4_common as P4
    from mtah import grumodel as G
    from mtah.deadball import features_v2
    from mtah.degrade import degrade, rates_for
    from mtah.external import from_soccertrack
    from p9b_oof_sweeps import HI, LO
    from p9ck_st_dense_fusion import evidence
    from p9cs_st_video_dead import TEST, video_dead
    from p9cv_evidence_shape import box
    os.makedirs(D, exist_ok=True)
    clf = P4.dead_classifier(featurize=features_v2, extra_rates=P4.CARRY_RATES, draws=P4.DRAWS_V6, metrica=(1, 2, 3), xmask=True)
    gm = P4.dead_gru(draws=P4.DRAWS_V6, metrica=(1, 2, 3), xmask=True, nseeds=6)
    for model, d in (("A", dA), ("B", dB)):
        for mid in TEST[model]:
            m0 = from_soccertrack(mid, merge_s=1.6)
            m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
            z = np.load(os.path.join(ROOT, "cache", "st_chain", "{}.npz".format(mid)))
            pt, sw, lab, g = z["p"].astype(np.float64), z["switch"], z["labels"], z["g"]
            F = features_v2(m)
            p = np.zeros(len(g))
            p[g] = clf.predict_proba(F[g])[:, 1]
            q = np.where(g, np.mean([G._predict_one(n, gm.norm(F.values.astype(np.float32)), g, gm.dev) for n in gm.nets], axis=0), 0)
            v = video_dead(d, mid, m0["half"])
            v = np.where(np.isnan(v), (p + q) / 2, v)
            lo = np.log(np.clip(pt, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - pt, 1e-4, 1 - 1e-4))
            pv = 1 / (1 + np.exp(-(lo + 0.5 * box(evidence(d, mid, m0["half"]), 151))))
            np.savez_compressed(os.path.join(D, "in_{}.npz".format(mid)), team0=pt, team=pv, dead2=(p + q) / 2, dead3=(p + q + v) / 3, sw=sw, lab=lab, g=g,
                                b=np.clip((HI - np.asarray(sw, np.float64)) / (HI - LO), 0, 1))
            print("prepared", mid, flush=True)


def decode(code):
    import copy
    sys.path.insert(0, os.path.join(code, "src"))
    from mtah import config
    from mtah.decoding import hsmm
    cfg = copy.deepcopy(config.load("main_ext_met_dt"))
    cfg["decoder"].update(CFG)
    for f in sorted(glob.glob(os.path.join(D, "in_*.npz"))):
        z = np.load(f)
        g = z["g"]
        e = np.diff(np.r_[0, g.astype(np.int8), 0])
        out = {}
        for name, team, dead in (("chain", "team0", "dead2"), ("video team", "team", "dead2"), ("video team + dead", "team", "dead3")):
            p, pd = z[team].astype(np.float64), z[dead].astype(np.float64)
            P = np.stack([(1 - pd) * p, (1 - pd) * (1 - p), pd], 1)
            lab = np.full(len(g), 3, np.int8)
            for a, c in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
                lab[a:c] = hsmm.decode_run(hsmm.emission_logprob(P[a:c], cfg), cfg, boundary=z["b"][a:c].astype(np.float64))
            out[name] = lab
        np.savez_compressed(f.replace("in_", "out_"), **out)
        print("decoded", os.path.basename(f), flush=True)


def score():
    sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
    import possession_metrics as pm
    from m1_dead_classifier import to_blocks
    from mtah.teammodel import decode as our_decode
    from p9l_rescore_calibrated import merged
    R = {}
    for f in sorted(glob.glob(os.path.join(D, "out_*.npz"))):
        z, o = np.load(f.replace("out_", "in_")), np.load(f)
        g, lab, sw = z["g"], z["lab"], z["sw"]
        l3 = merged(lab, 3.0)
        sc = lambda c: np.array([pm.spell_scores(L, np.where(g, c, pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)])
        for name, team, dead in (("chain", "team0", "dead2"), ("video team", "team", "dead2"), ("video team + dead", "team", "dead3")):
            dd = to_blocks(z[dead]) & g
            R.setdefault(name, []).append((sc(our_decode(z[team].astype(np.float64), g & ~dd, sw)), sc(o[name])))
    print("arm | our decoder 1.6 s / 3 s | HSMM | gain (matches ahead at 1.6 s, worst)")
    for k, v in R.items():
        a, b = np.array([x[0] for x in v]), np.array([x[1] for x in v])
        print("{} | {:.4f} / {:.4f} | {:.4f} / {:.4f} | {:+.4f} / {:+.4f} ({} / {}, {:+.4f})".format(k, *a.mean(0), *b.mean(0), *(b - a).mean(0), int(((b - a)[:, 0] > 0).sum()), len(a), (b - a).min()))


if __name__ == "__main__":
    {"prep": prep, "decode": decode, "score": score}[sys.argv[1]](*sys.argv[2:])
