"""P9cw: P9cv's smoothed video evidence scored with SoccerTrack's spells reweighted to DFL's length distribution.
SoccerTrack's label spells are long (P9cv) and ours are not, so its plain spellF1 over-rewards fragment removal. Here
every label spell and every predicted spell carries the weight share_DFL(bin) / share_SoccerTrack(bin) of its length
bin (label spells of both sets; bins < 2, 2-5, 5-10, 10-20, >= 20 s); weighted recall over label spells, weighted
precision over predicted spells, F1 per IoU, mean over the metric's IoUs. External data only.

    python analysis/p9cw_length_weighted.py <fold A predictions> <fold B predictions>
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
from p9l_rescore_calibrated import merged  # noqa: E402

EDGES = np.array([2, 5, 10, 20]) * 25
CONF = [(0, 1)] + [(w, s) for w in (0.5, 0.75) for s in (1, 25, 51, 101, 151)]


def bins(spells):
    return np.searchsorted(EDGES, [b - a for a, b, _ in spells], side="right")


def wscore(lab, c, wt):
    runs = pm.valid_runs(lab)
    gs, ps = pm.possession_spells(lab, runs), pm.possession_spells(c, runs)
    ov, wg, wp, f = pm.spell_overlaps(gs, ps), wt[bins(gs)], wt[bins(ps)], []
    for th in pm.SPELL_IOUS:
        mg, mp = np.zeros(len(gs), bool), np.zeros(len(ps), bool)
        for iou, i, j in ov:
            if iou < th:
                break
            if not mg[i] and not mp[j]:
                mg[i] = mp[j] = True
        r, p = wg[mg].sum() / max(wg.sum(), 1e-9), wp[mp].sum() / max(wp.sum(), 1e-9)
        f.append(2 * p * r / max(p + r, 1e-9))
    return float(np.mean(f))


def main(dA, dB):
    z = np.load(os.path.join(ROOT, "cache", "dfl_labels.npz"))
    dfl = np.bincount(np.concatenate([bins(pm.possession_spells(z[k], pm.valid_runs(z[k]))) for k in z.files]), minlength=5)
    data, st = [], np.zeros(5)
    for model, d in (("A", dA), ("B", dB)):
        for mid in TEST[model]:
            c = np.load(os.path.join(ROOT, "cache", "st_chain", "{}.npz".format(mid)))
            lab = c["labels"]
            st += np.bincount(bins(pm.possession_spells(lab, pm.valid_runs(lab))), minlength=5)
            data.append((mid, c["p"].astype(np.float64), c["switch"], lab, c["g"], evidence(d, mid, from_soccertrack(mid)["half"])))
    wt = (dfl / dfl.sum()) / (st / st.sum())
    print("length shares < 2 / 2-5 / 5-10 / 10-20 / >= 20 s: DFL", np.round(dfl / dfl.sum(), 3), "SoccerTrack", np.round(st / st.sum(), 3), "weights", np.round(wt, 2))
    S = np.zeros((len(data), len(CONF), 2))
    for m, (mid, p, sw, lab, g, e) in enumerate(data):
        alive = g & np.isin(lab, (pm.H, pm.A))
        lo = np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))
        for i, (w, s) in enumerate(CONF):
            c = np.where(g, np.where(alive, decode(1 / (1 + np.exp(-(lo + w * box(e, s)))), alive, sw), pm.D), pm.X).astype(np.int8)
            S[m, i] = wscore(lab, c, wt), wscore(lab, c, np.ones(5))
        print(mid, "done", flush=True)
    G = S[:, 1:] - S[:, :1]
    print("chain: weighted {:.4f} plain {:.4f}".format(*S[:, 0].mean(0)))
    print("w smooth | weighted gain | plain gain | ahead (weighted) | worst (weighted)")
    for i, (w, s) in enumerate(CONF[1:]):
        print("{:.2f} {:3d} | {:+.4f} | {:+.4f} | {} / 8 | {:+.4f}".format(w, s, *G[:, i].mean(0), int((G[:, i, 0] > 0).sum()), G[:, i, 0].min()))


if __name__ == "__main__":
    main(*sys.argv[1:3])
