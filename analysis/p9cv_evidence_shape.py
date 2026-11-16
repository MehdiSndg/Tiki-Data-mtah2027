"""P9cv: why the dense video evidence loses beyond w = 0.5 (P9cp sweep, 12 Nov): the shape of the evidence. On the 8
SoccerTrack matches, cross-fold as P9cp: the log ratio clipped at +-c and / or box-smoothed over s frames before the
fusion, over w. Mean gain at 1.6 s / 3 s, matches ahead, worst match. External data only; a pick from this table needs
the nested (leave-one-match-out) row printed at the end.

    python analysis/p9cv_evidence_shape.py <fold A predictions> <fold B predictions>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah.external import from_soccertrack  # noqa: E402
from p9cf_st_evidence_forms import score  # noqa: E402
from p9ck_st_dense_fusion import evidence  # noqa: E402
from p9cp_gateB import TEST  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

CONF = [(c, s, w) for c in (2.0, 99.0) for s in (1, 51, 101, 151, 251) for w in (0.5, 0.75, 1.0)]


def box(e, s):
    return e if s == 1 else np.convolve(e, np.ones(s) / s, "same")


def main(dA, dB):
    G, mids = [], []
    for model, d in (("A", dA), ("B", dB)):
        for mid in TEST[model]:
            z = np.load(os.path.join(ROOT, "cache", "st_chain", "{}.npz".format(mid)))
            p, sw, lab, g = z["p"].astype(np.float64), z["switch"], z["labels"], z["g"]
            e = evidence(d, mid, from_soccertrack(mid)["half"])
            alive = g & np.isin(lab, (pm.H, pm.A))
            l3 = merged(lab, 3.0)
            lo = np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))
            base = score(lab, l3, p, alive, sw, g)
            G.append([score(lab, l3, 1 / (1 + np.exp(-(lo + w * box(np.clip(e, -c, c), s)))), alive, sw, g) - base for c, s, w in CONF])
            mids.append(mid)
            print(mid, "done", flush=True)
    G = np.array(G)                                                  # match, config, (1.6 s, 3 s)
    print("clip smooth w | mean gain 1.6 s 3 s | ahead | worst")
    for i, (c, s, w) in enumerate(CONF):
        print("{:4.0f} {:3d} {:.1f} | {:+.4f} {:+.4f} | {} / 8 | {:+.4f}".format(c, s, w, *G[:, i].mean(0), int((G[:, i, 0] > 0).sum()), G[:, i].min()))
    nested = np.array([G[m, np.delete(G, m, 0).mean((0, 2)).argmax()] for m in range(len(G))])
    print("nested pick: mean gain {:+.4f} {:+.4f}, ahead {} / 8, worst {:+.4f}; w = 0.5 unshaped {:+.4f} {:+.4f}".format(
        *nested.mean(0), int((nested[:, 0] > 0).sum()), nested.min(), *G[:, CONF.index((99.0, 1, 0.5))].mean(0)))
    np.save(os.path.join(ROOT, "cache", "p9cv_gains.npy"), G)


if __name__ == "__main__":
    main(*sys.argv[1:3])
