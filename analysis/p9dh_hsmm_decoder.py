"""P9dh: Mali's explicit-duration semi-Markov decoder (cache/team_in/mehmet_ali_topkara/code, src/mtah/decoding/hsmm.py,
his externally tuned configuration main_ext_met_dt) on OUR out-of-fold probabilities, against our decoder (switch-cost
Viterbi on the team call inside dead-ball blocks). DFL 7 + Metrica 3, our labels, 1.6 s / 3 s.
Three steps, because his package is also called mtah and needs numba:
    python analysis/p9dh_hsmm_decoder.py prep                       # -> cache/p9dh/in_<match>.npz (P(H, A, D), boundary, halves)
    <venv with numpy, numba, pyyaml>/bin/python analysis/p9dh_hsmm_decoder.py decode <his code dir>   # -> cache/p9dh/out_<match>.npz
    python analysis/p9dh_hsmm_decoder.py score
Arms of the decode step: emission scale 1 / 2.5 (his) / 4, with and without our boundary probability as his boundary
bonus (gamma 0 / 2).
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "cache", "p9dh")
ARMS = [(s, gm) for s in (1.0, 2.5, 4.0) for gm in (0.0, 2.0)]
GRID = [(s, gm, pt, al) for s in (3.0, 4.0, 5.0, 6.0) for gm in (0.0, 2.0, 4.0) for pt in (6.0, 8.0, 10.0) for al in (2.0, 4.0)]   # scale, boundary gamma, turnover penalty, duration alpha
TAG = os.environ.get("GRID", "grid")
if TAG == "grid2":                                               # the first grid's best were at its edges (gamma 4, alpha 2)
    GRID = [(s, gm, pt, al) for s in (3.0, 4.0) for gm in (4.0, 6.0, 8.0) for pt in (6.0, 10.0) for al in (1.0, 2.0)]


def prep():
    sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
    C = lambda *p: os.path.join(ROOT, "cache", *p)
    os.makedirs(D, exist_ok=True)
    for f in sorted(glob.glob(C("oof_tcn128d3", "*.npz"))):
        n = os.path.basename(f)
        z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f))
        p = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
        pd = np.nan_to_num((z["qh"] + z["qg"]) / 2)
        P = np.stack([(1 - pd) * p, (1 - pd) * (1 - p), pd], 1)
        np.savez_compressed(os.path.join(D, "in_" + n[4:]), P=P.astype(np.float32), g=z["g"], b=np.nan_to_num(z["b"]).astype(np.float32))
        print("prepared", n[4:-4])


def decode(code):
    import copy
    sys.path.insert(0, os.path.join(code, "src"))
    from mtah import config
    from mtah.decoding import hsmm
    base = config.load("main_ext_met_dt")
    for f in sorted(glob.glob(os.path.join(D, "in_*.npz"))):
        z = np.load(f)
        P, g, b = z["P"].astype(np.float64), z["g"], z["b"].astype(np.float64)
        e = np.diff(np.r_[0, g.astype(np.int8), 0])
        runs = list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))
        out = {}
        for s, gm in ARMS:
            cfg = copy.deepcopy(base)
            cfg["decoder"]["emission_scale"] = s
            cfg["decoder"]["boundary_gamma"] = gm
            lab = np.full(len(g), 3, np.int8)
            for a, c in runs:
                lab[a:c] = hsmm.decode_run(hsmm.emission_logprob(P[a:c], cfg), cfg, boundary=b[a:c] if gm else None)
            out["s{}_g{}".format(s, gm)] = lab
        np.savez_compressed(f.replace("in_", "out_"), **out)
        print("decoded", os.path.basename(f), flush=True)


def decodegrid(code):
    import copy
    sys.path.insert(0, os.path.join(code, "src"))
    from mtah import config
    from mtah.decoding import hsmm
    base = config.load("main_ext_met_dt")
    for f in sorted(glob.glob(os.path.join(D, "in_*.npz"))):
        dst = f.replace("in_", TAG + "_")
        if os.path.exists(dst):
            continue
        z = np.load(f)
        P, g, b = z["P"].astype(np.float64), z["g"], z["b"].astype(np.float64)
        e = np.diff(np.r_[0, g.astype(np.int8), 0])
        runs = list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))
        out = np.full((len(GRID), len(g)), 3, np.int8)
        for i, (s, gm, pt, al) in enumerate(GRID):
            cfg = copy.deepcopy(base)
            cfg["decoder"].update(emission_scale=s, boundary_gamma=gm, penalty_turnover=pt, alpha_duration=al)
            for a, c in runs:
                out[i, a:c] = hsmm.decode_run(hsmm.emission_logprob(P[a:c], cfg), cfg, boundary=b[a:c] if gm else None)
        np.savez_compressed(dst, lab=out)
        print("grid decoded", os.path.basename(f), flush=True)


def scoregrid():
    """Nested: each match with the configuration that is best on the other nine (mean of both referees)."""
    sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
    import possession_metrics as pm
    from m1_dead_classifier import to_blocks
    from mtah.teammodel import decode as our_decode
    from p9b_oof_sweeps import HI, LO
    from p9l_rescore_calibrated import merged
    C = lambda *p: os.path.join(ROOT, "cache", *p)
    S, B, names = [], [], []
    for f in sorted(glob.glob(os.path.join(D, TAG + "_*.npz"))):
        n = "oof_" + os.path.basename(f)[len(TAG) + 1:]
        z = dict(np.load(C("oof_v16", n), allow_pickle=True))
        g, lab, P = z["g"], z["labels"], np.load(f.replace(TAG + "_", "in_"))["P"].astype(np.float64)
        l3 = merged(lab, 3.0)
        sc = lambda c: [pm.spell_scores(L, np.where(g, c, pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        p, dead = P[:, 0] / np.maximum(P[:, 0] + P[:, 1], 1e-9), to_blocks(P[:, 2]) & g
        B.append(sc(our_decode(p, g & ~dead, HI * (1 - z["b"]) + LO * z["b"])))
        S.append([sc(c) for c in np.load(f)["lab"]])
        names.append(n[4:-4])
    S, B = np.array(S), np.array(B)                                  # match, config, referee
    G = S - B[:, None]
    dfl = np.array(["DFL" in n for n in names])
    order = np.argsort(-G.mean((0, 2)))
    print("scale gamma turnover alpha | DFL gain 1.6 s / 3 s (ahead) | Metrica gain 1.6 s / 3 s (ahead)")
    for i in order[:12]:
        print("{:.0f} {:.0f} {:.0f} {:.0f} | {:+.4f} / {:+.4f} ({} / 7) | {:+.4f} / {:+.4f} ({} / 3)".format(*GRID[i], *G[dfl, i].mean(0), int((G[dfl, i, 0] > 0).sum()), *G[~dfl, i].mean(0), int((G[~dfl, i, 0] > 0).sum())))
    nested = np.array([G[m, np.delete(G, m, 0).mean((0, 2)).argmax()] for m in range(len(G))])
    picks = [GRID[np.delete(G, m, 0).mean((0, 2)).argmax()] for m in range(len(G))]
    print("nested pick: DFL {:+.4f} / {:+.4f} ({} / 7), Metrica {:+.4f} / {:+.4f} ({} / 3); picks {}".format(*nested[dfl].mean(0), int((nested[dfl, 0] > 0).sum()), *nested[~dfl].mean(0), int((nested[~dfl, 0] > 0).sum()), sorted(set(picks))))


def score():
    sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
    import possession_metrics as pm
    from m1_dead_classifier import to_blocks
    from mtah.teammodel import decode as our_decode
    from p9b_oof_sweeps import HI, LO
    from p9l_rescore_calibrated import merged
    C = lambda *p: os.path.join(ROOT, "cache", *p)
    rows = {}
    for f in sorted(glob.glob(os.path.join(D, "out_*.npz"))):
        n = "oof_" + os.path.basename(f)[4:]
        z = dict(np.load(C("oof_v16", n), allow_pickle=True))
        i, o = np.load(f.replace("out_", "in_")), np.load(f)
        g, lab = z["g"], z["labels"]
        l3 = merged(lab, 3.0)
        sc = lambda c: np.array([pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)])
        P = i["P"].astype(np.float64)
        p, dead = P[:, 0] / np.maximum(P[:, 0] + P[:, 1], 1e-9), to_blocks(P[:, 2]) & g
        r = {"ours": sc(np.where(g, our_decode(p, g & ~dead, HI * (1 - z["b"]) + LO * z["b"]), pm.X).astype(np.int8))}
        for k in o.files:
            c = o[k].astype(np.int8)
            r["hsmm " + k] = sc(np.where(g, c, pm.X).astype(np.int8))
            r["hsmm " + k + ", our dead blocks"] = sc(np.where(g, np.where(dead, pm.D, np.where(c == pm.D, np.where(p > 0.5, pm.H, pm.A), c)), pm.X).astype(np.int8))
        for k, v in r.items():
            rows.setdefault(k, {})[n[4:-4]] = v
    base = rows["ours"]
    print("arm | DFL 1.6 s / 3 s | gain (ahead) | Metrica 1.6 s / 3 s | gain (ahead)")
    for k, v in rows.items():
        out = [k]
        for src in ("DFL", "metrica"):
            ms = [m for m in v if src in m]
            a, b = np.array([v[m] for m in ms]), np.array([base[m] for m in ms])
            out.append("{:.4f} / {:.4f} | {:+.4f} / {:+.4f} ({} / {})".format(*a.mean(0), *(a - b).mean(0), int(((a - b)[:, 0] > 0).sum()), len(ms)))
        print(" | ".join(out))


if __name__ == "__main__":
    {"prep": prep, "decode": decode, "score": score, "decodegrid": decodegrid, "scoregrid": scoregrid}[sys.argv[1]](*sys.argv[2:])
