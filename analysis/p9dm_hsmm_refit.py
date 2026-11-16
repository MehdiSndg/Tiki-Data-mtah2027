"""P9dm: does the semi-Markov decoder's setting move once Mali's model is in the probabilities (v26's fusion: team
0.75 ours + 0.25 his in log-odds, dead = mean of our two models and his)? The second grid of P9dh (scale 3 / 4, boundary
gamma 4 / 6 / 8, turnover 6 / 10, duration weight 1 / 2) on the fused probabilities, DFL 7 + Metrica 3, with a
leave-one-match-out pick, against v26's fixed setting (4 / 6 / 10 / 2). CPU.

    python analysis/p9dm_hsmm_refit.py
"""
import copy
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import semimarkov  # noqa: E402
from p9de_team_fuse import C, lg, sg, theirs  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

GRID = [(4.0, 6.0, 10.0, 2.0)] + [(s, gm, pt, al) for s in (3.0, 4.0, 5.0) for gm in (4.0, 6.0, 8.0) for pt in (6.0, 10.0) for al in (1.0, 2.0) if (s, gm, pt, al) != (4.0, 6.0, 10.0, 2.0)]


def main():
    S, names = [], []
    for f in sorted(glob.glob(C("oof_tcn128d3", "*.npz"))):
        n = os.path.basename(f)
        z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f))
        g, lab, name = z["g"], z["labels"], n[4:-4]
        M = theirs("mehmet_ali_topkara", name, g)
        l3, b = merged(lab, 3.0), np.nan_to_num(z["b"])
        p = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
        team, dead = sg(0.75 * lg(p) + 0.25 * lg(M["team"])), (np.nan_to_num(z["qh"]) + np.nan_to_num(z["qg"]) + M["p_dead"]) / 3
        row = []
        for s, gm, pt, al in GRID:
            cfg = copy.deepcopy(semimarkov.CONFIG)
            cfg["decoder"].update(emission_scale=s, boundary_gamma=gm, penalty_turnover=pt, alpha_duration=al)
            c = np.where(g, semimarkov.decode_halves(team, dead, g, b, cfg=cfg), pm.X).astype(np.int8)
            row.append([pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)])
        S.append(row)
        names.append(name)
        print(name, "done", flush=True)
    S = np.array(S)
    G = S - S[:, :1]
    dfl = np.array(["DFL" in n for n in names])
    print("fixed setting: DFL {:.4f} / {:.4f}, Metrica {:.4f} / {:.4f}".format(*S[dfl, 0].mean(0), *S[~dfl, 0].mean(0)))
    print("scale gamma turnover weight | DFL gain over the fixed setting (ahead) | Metrica gain (ahead)")
    for i in np.argsort(-G.mean((0, 2)))[:8]:
        print("{:.0f} {:.0f} {:.0f} {:.0f} | {:+.4f} / {:+.4f} ({} / 7) | {:+.4f} / {:+.4f} ({} / 3)".format(*GRID[i], *G[dfl, i].mean(0), int((G[dfl, i, 0] > 0).sum()), *G[~dfl, i].mean(0), int((G[~dfl, i, 0] > 0).sum())))
    nested = np.array([G[m, np.delete(G, m, 0).mean((0, 2)).argmax()] for m in range(len(G))])
    print("nested pick: DFL {:+.4f} / {:+.4f} ({} / 7), Metrica {:+.4f} / {:+.4f} ({} / 3)".format(*nested[dfl].mean(0), int((nested[dfl, 0] > 0).sum()), *nested[~dfl].mean(0), int((nested[~dfl, 0] > 0).sum())))


if __name__ == "__main__":
    main()
