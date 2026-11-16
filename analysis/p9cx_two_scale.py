"""P9cx: raw and smoothed video evidence together (P9cv / P9cw: the raw form helps short spells, the 151-frame box
removes fragments): w_raw * e + w_s * box(e, s). SoccerTrack cross-fold, plain spellF1 at 1.6 s and DFL-length-weighted
(P9cw), with a leave-one-match-out pick. External data only.

    python analysis/p9cx_two_scale.py <fold A predictions> <fold B predictions>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah.external import from_soccertrack  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p9ck_st_dense_fusion import evidence  # noqa: E402
from p9cp_gateB import TEST  # noqa: E402
from p9cv_evidence_shape import box  # noqa: E402
from p9cw_length_weighted import bins, wscore  # noqa: E402

CONF = [(0, 0, 1)] + [(wr, ws, s) for s in (151, 251) for wr in (0, 0.25, 0.5) for ws in (0.5, 0.75, 1.0, 1.5)]


def main(dA, dB):
    z = np.load(os.path.join(ROOT, "cache", "dfl_labels.npz"))
    dfl = np.bincount(np.concatenate([bins(pm.possession_spells(z[k], pm.valid_runs(z[k]))) for k in z.files]), minlength=5)
    data, st = [], np.zeros(5)
    for model, d in (("A", dA), ("B", dB)):
        for mid in TEST[model]:
            c = np.load(os.path.join(ROOT, "cache", "st_chain", "{}.npz".format(mid)))
            st += np.bincount(bins(pm.possession_spells(c["labels"], pm.valid_runs(c["labels"]))), minlength=5)
            data.append((c["p"].astype(np.float64), c["switch"], c["labels"], c["g"], evidence(d, mid, from_soccertrack(mid)["half"])))
    wt = (dfl / dfl.sum()) / (st / st.sum())
    S = np.zeros((len(data), len(CONF), 2))
    for m, (p, sw, lab, g, e) in enumerate(data):
        alive = g & np.isin(lab, (pm.H, pm.A))
        lo = np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))
        for i, (wr, ws, s) in enumerate(CONF):
            c = np.where(g, np.where(alive, decode(1 / (1 + np.exp(-(lo + wr * e + ws * box(e, s)))), alive, sw), pm.D), pm.X).astype(np.int8)
            S[m, i] = wscore(lab, c, wt), wscore(lab, c, np.ones(5))
        print(m, "done", flush=True)
    G = S[:, 1:] - S[:, :1]
    print("w_raw w_s smooth | weighted gain | plain gain | ahead w / p | worst w / p")
    for i, (wr, ws, s) in enumerate(CONF[1:]):
        print("{:.2f} {:.2f} {:3d} | {:+.4f} | {:+.4f} | {} {} | {:+.4f} {:+.4f}".format(wr, ws, s, *G[:, i].mean(0), *(G[:, i] > 0).sum(0), *G[:, i].min(0)))
    nested = np.array([G[m, np.delete(G, m, 0).mean((0, 2)).argmax()] for m in range(len(G))])
    print("nested pick: weighted {:+.4f} plain {:+.4f}, ahead {} {}, worst {:+.4f} {:+.4f}".format(*nested.mean(0), *(nested > 0).sum(0), *nested.min(0)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
