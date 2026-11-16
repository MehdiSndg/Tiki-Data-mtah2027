"""P9cq: action evidence on top of the dense video evidence (rule registered in notes/BULGULAR.md on 11 Nov before the
run): the cross-fold predictions of P9ci on the 8 SoccerTrack matches; detections = peaks of 1 - P(background) >= 0.6
with a +-0.5 s suppression whose class is a possession action, team from T > 0.5; pair form of P9cf with w_a = 1.0
between two consecutive detections of one team at most 4 s apart; added to chain + 0.5 x dense evidence.

    python analysis/p9cq_st_action_on_dense.py <fold A model's predictions> <fold B model's predictions>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah.external import SOCCERTRACK, from_soccertrack, soccertrack_side_codes  # noqa: E402
from p9cf_st_evidence_forms import fuse, score  # noqa: E402
from p9cg_tdeed_eval import peaks  # noqa: E402
from p9ck_st_dense_fusion import evidence  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

TEST = {"A": ("118577", "118578", "128057", "128058"), "B": ("117092", "117093", "118575", "118576")}
POSS_IDX = (1, 2, 4, 6, 7, 8, 11)          # PASS, DRIVE, HIGH PASS, CROSS, THROW IN, SHOT, FREE KICK in P9bz's class order (+1)
THR, W_D, W_A = 0.6, 0.5, 1.0


def detections(d, mid, half):
    """[(frame in the concatenated match, +1 home / -1 away)], sorted."""
    acts = json.load(open(os.path.join(SOCCERTRACK, "bas", mid, "{}_12_class_events.json".format(mid))))["actions"]
    n1 = int(np.sum(half == 1))
    out = []
    for h, hname in ((1, "1st"), (2, "2nd")):
        z = np.load(os.path.join(d, "pred_{}_{}.npz".format(mid, hname)))
        P, T = z["P"], z["T"]
        st = np.load(os.path.join(SOCCERTRACK, "ball", "{}_{}_ball.npz".format(mid, hname)))["status"]
        codes = soccertrack_side_codes([a for a in acts if a["gameTime"].strip().startswith("{} ".format(h))], st)
        left_home = codes["left"] == 1
        nh = int(np.sum(half == h))
        for i in peaks(1 - P[:, 0], THR):
            if int(np.argmax(P[i, 1:])) + 1 not in POSS_IDX or 2 * i >= nh:
                continue
            right = T[i] > 0.5
            out.append((2 * i + (0 if h == 1 else n1 + 1), 1 if (right != left_home) else -1))
    return sorted(out)


def main(dA, dB):
    G = []
    for model, d in (("A", dA), ("B", dB)):
        for mid in TEST[model]:
            z = np.load(os.path.join(ROOT, "cache", "st_chain", "{}.npz".format(mid)))
            p, sw, lab, g = z["p"].astype(np.float64), z["switch"], z["labels"], z["g"]
            half = from_soccertrack(mid)["half"]
            alive = g & np.isin(lab, (pm.H, pm.A))
            l3 = merged(lab, 3.0)
            lo = np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4)) + W_D * evidence(d, mid, half)
            ev = detections(d, mid, half)
            base = score(lab, l3, 1 / (1 + np.exp(-lo)), alive, sw, g)
            q, s = fuse("pair", W_A, ev, lo, sw)
            new = score(lab, l3, q, alive, s, g)
            G.append(new - base)
            print(mid, "detections", len(ev), "dense {:.4f} {:.4f} + actions {:.4f} {:.4f} gain {:+.4f} {:+.4f}".format(*base, *new, *(new - base)), flush=True)
    G = np.array(G)
    ok = G.mean(0).min() >= 0.005 and (G[:, 0] > 0).sum() >= 6 and G.min() >= -0.01
    print("mean gain {:+.4f} {:+.4f}, ahead {} / {}, worst {:+.4f} -> {}".format(*G.mean(0), int((G[:, 0] > 0).sum()), len(G), G.min(), "HOLDS" if ok else "does not hold"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
