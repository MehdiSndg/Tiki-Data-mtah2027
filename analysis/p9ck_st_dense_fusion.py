"""P9ck: the decision test of the dense video possession head (notes/BULGULAR.md, 10 Nov, registered before any
result): P9ci's per-frame left / right / dead probabilities on SoccerTrack v2 test matches, fused into v19c's team call
of P9bw (cache/st_chain) as w * (log q_home - log q_away) added to the log-odds of every frame, w = 0.5 fixed in advance;
dead ball from the labels in both arms. Prints spellF1 at 1.6 s and 3 s per match with and without the evidence, the
evidence's own team accuracy on live frames and its agreement with the chain.

    python analysis/p9ck_st_dense_fusion.py <dir with pred_<match>_<half>.npz> [w]
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah.external import SOCCERTRACK, from_soccertrack, soccertrack_side_codes  # noqa: E402
from p9cf_st_evidence_forms import score  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402


def evidence(d, mid, half):
    """(n,) log q_home - log q_away per frame of the concatenated match; 0 where there is no prediction."""
    acts = json.load(open(os.path.join(SOCCERTRACK, "bas", mid, "{}_12_class_events.json".format(mid))))["actions"]
    n1 = int(np.sum(half == 1))
    e = np.zeros(len(half))
    for h, hname in ((1, "1st"), (2, "2nd")):
        path = os.path.join(d, "pred_{}_{}.npz".format(mid, hname))
        if not os.path.exists(path):
            continue
        st = np.load(os.path.join(SOCCERTRACK, "ball", "{}_{}_ball.npz".format(mid, hname)))["status"]
        codes = soccertrack_side_codes([a for a in acts if a["gameTime"].strip().startswith("{} ".format(h))], st)
        Q = np.clip(np.repeat(np.load(path)["Q"].astype(np.float64), 2, axis=0), 1e-4, 1)
        lr = np.log(Q[:, 0]) - np.log(Q[:, 1])                      # left over right
        if codes["left"] != 1:                                      # code 1 is home
            lr = -lr
        off = 0 if h == 1 else n1 + 1
        n = min(len(lr), int(np.sum(half == h)))
        e[off:off + n] = lr[:n]
    return e


def main(d, w=0.5):
    mids = sorted({os.path.basename(p)[5:-4].split("_")[0] for p in glob.glob(os.path.join(d, "pred_*.npz"))})
    gains = []
    for mid in mids:
        z = np.load(os.path.join(ROOT, "cache", "st_chain", "{}.npz".format(mid)))
        p, sw, lab, g = z["p"].astype(np.float64), z["switch"], z["labels"], z["g"]
        e = evidence(d, mid, from_soccertrack(mid)["half"])
        alive = g & np.isin(lab, (pm.H, pm.A))
        l3 = merged(lab, 3.0)
        lo = np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))
        base = score(lab, l3, p, alive, sw, g)
        new = score(lab, l3, 1 / (1 + np.exp(-(lo + w * e))), alive, sw, g)
        home = lab[alive] == pm.H
        print(mid, "base {:.4f} {:.4f} with evidence {:.4f} {:.4f} gain {:+.4f} {:+.4f}".format(*base, *new, *(new - base)),
              "| team accuracy on live frames: evidence {:.3f} chain {:.3f} fused {:.3f}, evidence agrees with chain {:.3f}".format(
                  np.mean((e[alive] > 0) == home), np.mean((lo[alive] > 0) == home), np.mean((lo[alive] + w * e[alive] > 0) == home),
                  np.mean((e[alive] > 0) == (lo[alive] > 0))), flush=True)
        gains.append(new - base)
    gm = np.mean(gains, axis=0)
    print("w", w, "mean gain {:+.4f} {:+.4f}".format(*gm), "matches ahead", int(np.sum(np.array(gains)[:, 0] > 0)), "/", len(gains),
          "-> CONTINUE" if gm.min() >= 0.02 and np.all(np.array(gains) > 0) else "-> STOP")


if __name__ == "__main__":
    main(sys.argv[1], *(float(x) for x in sys.argv[2:3]))
