"""P9dz: why the mix-trained dead-ball GRU collapses on PFF (P9dx: 0.8064 -> 0.6007). Same fits as P9dx on all ten
matches; per PFF match: true and predicted dead share, dead precision / recall, mean probability on true dead / live
frames, and the same with PFF's ball restricted to its seen frames (ball_vis). GPU.

    .venv/bin/python analysis/p9dz_gru_pff_diag.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p9dx_deadgru_mix as X  # noqa: E402
from p9dx_deadgru_mix import G, P4, feats, pff_ids, pm, to_blocks, DFL, glob  # noqa: E402


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    names = ids + ["metrica{}".format(g) for g in (1, 2, 3)]
    nope = lambda: (_ for _ in ()).throw(RuntimeError("cache missing"))
    C = {(s, n, d): feats("{}_{}_{}".format(k, n, d), nope)[:2] for s, k in (("prod", "sim"), ("v6", "simv6")) for n in names for d in P4.DRAWS_V6}
    pff = {i: feats("pff_{}".format(i), nope) for i in pff_ids()[-12:]}
    cols = pff[next(iter(pff))][2]
    for arm, srcs in (("prod", ("prod",)), ("mix", ("prod", "v6"))):
        parts = [C[(s, n, d)] for s in srcs for n in names for d in P4.DRAWS_V6]
        allF = np.concatenate([F for F, _ in parts])
        mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
        data = [(mod.norm(F), (lab == pm.D).astype(np.float32), (lab != pm.X).astype(np.float32), lab != pm.X) for F, lab in parts]
        mod.nets = [G.train(data, s, "cuda") for s in G.SEEDS]
        print("ARM", arm, "train feature means (first 6 cols):", dict(zip(cols[:6], np.round(np.nanmean(allF, 0)[:6], 3))), flush=True)
        for i, (F, lab, _) in list(pff.items())[:6]:
            g = lab != pm.X
            q = np.where(g, np.mean([G._predict_one(n, mod.norm(F.astype(np.float32)), g, "cuda") for n in mod.nets], axis=0), 0)
            dead, truth = to_blocks(q) & g, (lab == pm.D) & g
            tp = np.sum(dead & truth)
            print(arm, i, "true {:.3f} pred {:.3f} prec {:.3f} rec {:.3f} | mean q on dead {:.3f} live {:.3f}".format(
                truth[g].mean(), dead[g].mean(), tp / max(dead.sum(), 1), tp / max(truth.sum(), 1), q[truth].mean(), q[g & ~truth].mean()), flush=True)
        print("PFF feature means (first 6 cols):", dict(zip(cols[:6], np.round(np.nanmean(np.concatenate([v[0] for v in pff.values()]), 0)[:6], 3))))


if __name__ == "__main__":
    main()
