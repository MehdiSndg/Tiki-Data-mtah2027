"""P9fm: P9fk's 81 decoder settings (scale, gamma, turnover, duration weight; D weight 1.0 as P9fi) scored on the second
judge too (DFL 7 + Metrica 3 out-of-fold, P9dm's inputs), so that a setting has to hold on both.
Registered before the run (16 Nov): a setting replaces production only if (1) its leave-one-match-out pick on SoccerTrack
(P9fk's scores, cache/p9fk_scores.npy) is >= +0.003 at both referees and >= 6/8, and (2) on DFL and on Metrica it is not
behind production by more than 0.002 at either referee; among the settings that pass (2), the SoccerTrack pick is nested
over that subset. Otherwise production (4, 6, 10, 2) stays.
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

GRID = list(itertools.product((3.0, 4.0, 5.0), (4.0, 6.0, 8.0), (6.0, 10.0, 14.0), (1.5, 2.0, 3.0)))   # = P9fk's order
PROD = (4.0, 6.0, 10.0, 2.0)


def load(n):
    from p9de_team_fuse import C, lg, sg, theirs
    from p9l_rescore_calibrated import merged
    z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), C("oof_tcn128d3", n)))
    g, lab, name = z["g"], z["labels"], n[4:-4]
    M = theirs("mehmet_ali_topkara", name, g)
    p = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
    team = sg(0.75 * lg(p) + 0.25 * lg(M["team"]))
    dead = (np.nan_to_num(z["qh"]) + np.nan_to_num(z["qg"]) + M["p_dead"]) / 3
    return g, lab, merged(lab, 3.0), team, dead, np.nan_to_num(z["b"])


def one(args):
    n, (s, gm, pt, al) = args
    from mtah import semimarkov
    g, lab, l3, team, dead, b = load(n)
    cfg = copy.deepcopy(semimarkov.CONFIG)
    cfg["decoder"].update(emission_scale=s, boundary_gamma=gm, penalty_turnover=pt, alpha_duration=al)
    cfg["decoder"]["emission_weights"][2] = 1.0
    c = np.where(g, semimarkov.decode_halves(team, dead, g, b, cfg=cfg), pm.X).astype(np.int8)
    return [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]


def main():
    from p9de_team_fuse import C
    names = sorted(os.path.basename(f) for f in glob.glob(C("oof_tcn128d3", "*.npz")))
    out = os.path.join(ROOT, "cache", "p9fm_scores.npy")
    if not os.path.exists(out):
        with Pool(3) as pool:
            r = pool.map(one, [(n, c) for c in GRID for n in names], chunksize=2)
        np.save(out, np.array(r).reshape(len(GRID), len(names), 2))
    D = np.load(out)                                   # config x match x referee
    dfl = np.array(["DFL" in n for n in names])
    base = GRID.index(PROD)
    gd, gm_ = (D[:, dfl] - D[base, dfl]).mean(1), (D[:, ~dfl] - D[base, ~dfl]).mean(1)
    ok2 = (gd >= -0.002).all(1) & (gm_ >= -0.002).all(1)
    S = np.load(os.path.join(ROOT, "cache", "p9fk_scores.npy"))      # match x config x referee (SoccerTrack)
    cand = np.flatnonzero(ok2)
    print("settings passing the second judge:", len(cand), "of", len(GRID))
    st_mean = S.mean(0)
    for i in cand[np.argsort(-st_mean[cand, 0])][:8]:
        print(GRID[i], "ST {:.4f} / {:.4f}".format(*st_mean[i]), "DFL {:+.4f} / {:+.4f}".format(*gd[i]), "Metrica {:+.4f} / {:+.4f}".format(*gm_[i]))
    nested = []
    for m in range(S.shape[0]):
        keep = [j for j in range(S.shape[0]) if j != m]
        k = cand[int(np.argmax(S[keep][:, cand].mean(axis=(0, 2))))]
        nested.append(S[m, k] - S[m, base])
    d = np.array(nested)
    ok = (d.mean(0) >= 0.003).all() and int((d[:, 0] > 0).sum()) >= 6
    print("SoccerTrack nested over the passing settings: {:+.4f} / {:+.4f} ({}/8) -> {}".format(*d.mean(0), int((d[:, 0] > 0).sum()), "holds" if ok else "FAILS"))


if __name__ == "__main__":
    main()
