"""P9fk's second judge: DFL 7 + Metrica 3 out-of-fold (P9dm's inputs: v26-era fusion, no video), production decoder
(4, 6, 10, 2) vs P9fk's (3, 8, 10, 3), each with D's emission weight 1.15 and 1.0 (P9fi).
Registered before the run (16 Nov): P9fk's setting stays adopted unless it is behind production by more than 0.005 at
either referee on DFL or on Metrica (this judge's labels are the furthest from the organisers', see the 16 Nov notes).
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

CFGS = {"prod": (4.0, 6.0, 10.0, 2.0, 1.15), "prod D1": (4.0, 6.0, 10.0, 2.0, 1.0),
        "P9fk": (3.0, 8.0, 10.0, 3.0, 1.15), "P9fk D1": (3.0, 8.0, 10.0, 3.0, 1.0)}


def main():
    rows = {}
    for f in sorted(glob.glob(C("oof_tcn128d3", "*.npz"))):
        n = os.path.basename(f)
        z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f))
        g, lab, name = z["g"], z["labels"], n[4:-4]
        M = theirs("mehmet_ali_topkara", name, g)
        l3, b = merged(lab, 3.0), np.nan_to_num(z["b"])
        p = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
        team, dead = sg(0.75 * lg(p) + 0.25 * lg(M["team"])), (np.nan_to_num(z["qh"]) + np.nan_to_num(z["qg"]) + M["p_dead"]) / 3
        r = {}
        for k, (s, gm, pt, al, ew) in CFGS.items():
            cfg = copy.deepcopy(semimarkov.CONFIG)
            cfg["decoder"].update(emission_scale=s, boundary_gamma=gm, penalty_turnover=pt, alpha_duration=al)
            cfg["decoder"]["emission_weights"][2] = ew
            c = np.where(g, semimarkov.decode_halves(team, dead, g, b, cfg=cfg), pm.X).astype(np.int8)
            r[k] = [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[name] = r
        print(name, {k: round(v[0], 4) for k, v in r.items()}, flush=True)
    for grp, sel in (("DFL", lambda n: "DFL" in n), ("Metrica", lambda n: "DFL" not in n)):
        ns = [n for n in rows if sel(n)]
        print(grp, {k: "{:.4f} / {:.4f}".format(*np.mean([rows[n][k] for n in ns], axis=0)) for k in CFGS})


if __name__ == "__main__":
    main()
