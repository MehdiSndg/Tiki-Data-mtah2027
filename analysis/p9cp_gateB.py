"""P9cp: gate B of the video evidence in full (notes/BULGULAR.md, 9 Nov and its amendments, all written before any video
result), on the SoccerTrack v2 matches with a possession status: each match with the model that never saw it (fold A's
model on fold B's matches and the reverse), w = 0.5 fixed (10 Nov), dead ball from the labels in both arms.
  mean gain at 1.6 s and at 3 s >= +0.020; >= 70 % of the halves ahead at 1.6 s; no half worse than -0.02;
  team accuracy on live frames >= 0.5 s away from every action: at most -0.2 points.

    python analysis/p9cp_gateB.py <dir with fold A's model's predictions> <dir with fold B's model's predictions> [w]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah.external import SOCCERTRACK, from_soccertrack  # noqa: E402
from p9cf_st_evidence_forms import score  # noqa: E402
from p9ck_st_dense_fusion import evidence  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

TEST = {"A": ("118577", "118578", "128057", "128058"), "B": ("117092", "117093", "118575", "118576")}   # model -> its test matches
FAR = 13


def main(dA, dB, w=0.5):
    M, Hf, far = [], [], np.zeros(3)
    for model, d in (("A", dA), ("B", dB)):
        for mid in TEST[model]:
            if not os.path.exists(os.path.join(d, "pred_{}_1st.npz".format(mid))):
                print(mid, "no predictions yet")
                continue
            z = np.load(os.path.join(ROOT, "cache", "st_chain", "{}.npz".format(mid)))
            p, sw, lab, g = z["p"].astype(np.float64), z["switch"], z["labels"], z["g"]
            half = from_soccertrack(mid)["half"]
            e = evidence(d, mid, half)
            alive = g & np.isin(lab, (pm.H, pm.A))
            l3 = merged(lab, 3.0)
            lo = np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))
            q = 1 / (1 + np.exp(-(lo + w * e)))
            gain = score(lab, l3, q, alive, sw, g) - score(lab, l3, p, alive, sw, g)
            M.append(gain)
            for h in (1, 2):
                k = half == h
                Hf.append((score(lab[k], l3[k], q[k], alive[k], sw[k], g[k]) - score(lab[k], l3[k], p[k], alive[k], sw[k], g[k]))[0])
            near = np.zeros(len(lab), bool)
            n1 = int(np.sum(half == 1))
            for a in json.load(open(os.path.join(SOCCERTRACK, "bas", mid, "{}_12_class_events.json".format(mid))))["actions"]:
                k = int(a["frame"]) + (0 if a["gameTime"].strip().startswith("1 ") else n1 + 1)
                near[max(k - FAR + 1, 0):k + FAR] = True
            f = alive & ~near
            home = lab[f] == pm.H
            far += (np.sum((lo[f] > 0) == home), np.sum((lo[f] + w * e[f] > 0) == home), f.sum())
            print(mid, "model", model, "gain {:+.4f} {:+.4f}".format(*gain), "halves {:+.4f} {:+.4f}".format(*Hf[-2:]), flush=True)
    M, Hf = np.array(M), np.array(Hf)
    d_far = 100 * (far[1] - far[0]) / far[2]
    ok = len(M) == 8 and M.mean(0).min() >= 0.02 and np.mean(Hf > 0) >= 0.7 and Hf.min() >= -0.02 and d_far >= -0.2
    print("matches {} mean gain {:+.4f} {:+.4f}; halves ahead at 1.6 s {} / {} ({:.0%}), worst half {:+.4f}; team accuracy away from actions "
          "chain {:.2%} fused {:.2%} ({:+.2f} points) -> gate B {}".format(len(M), *M.mean(0), int((Hf > 0).sum()), len(Hf), np.mean(Hf > 0), Hf.min(),
                                                                         far[0] / far[2], far[1] / far[2], d_far, "HOLDS" if ok else "does not hold / incomplete"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], *(float(x) for x in sys.argv[3:4])))
