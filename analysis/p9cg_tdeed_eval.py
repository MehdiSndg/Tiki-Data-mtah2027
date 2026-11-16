"""P9cg: detection quality of the T-DEED predictions of P9bz on SoccerTrack v2 halves (external data only): at several
score thresholds, recall and precision of the ball actions within +-1 s (peaks of 1 - P(background) with a +-0.5 s
non-maximum suppression, matched greedily in score order) and the accuracy of the team (left / right) at the matched
actions of the possession classes. The decision test of 10 Nov: recall >= 0.7 and team accuracy >= 0.9 at a precision
>= 0.7, or the video path stops.

    python analysis/p9cg_tdeed_eval.py <dir with pred_<match>_<half>.npz>
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BAS = os.path.join(ROOT, "data", "external", "soccertrack", "bas")
POSS = ("PASS", "DRIVE", "HIGH PASS", "CROSS", "SHOT", "THROW IN", "FREE KICK")
NMS, TOL = 6, 12                                        # frames at 12.5 fps


def peaks(s, thr):
    out = []
    for i in np.argsort(-s):
        if s[i] < thr:
            break
        if all(abs(i - j) > NMS for j in out):
            out.append(int(i))
    return out


def main(d):
    tot = {}
    for path in sorted(glob.glob(os.path.join(d, "pred_*.npz"))):
        mid, hname = os.path.basename(path)[5:-4].split("_")
        z = np.load(path)
        s, T = 1 - z["P"][:, 0], z["T"]
        h = 1 if hname == "1st" else 2
        acts = [a for a in json.load(open(os.path.join(BAS, mid, "{}_12_class_events.json".format(mid))))["actions"]
                if a["gameTime"].strip().startswith("{} ".format(h))]
        gt = [(int(a["frame"]) // 2, a["label"], a.get("team")) for a in acts]
        for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            pk = peaks(s, thr)
            used, hit, tok, tn = set(), 0, 0, 0
            for i in pk:
                c = [(abs(i - g[0]), k) for k, g in enumerate(gt) if k not in used and abs(i - g[0]) <= TOL]
                if not c:
                    continue
                k = min(c)[1]
                used.add(k)
                hit += 1
                if gt[k][1] in POSS and gt[k][2] in ("left", "right"):
                    tn += 1
                    tok += (T[i] > 0.5) == (gt[k][2] == "right")
            r = tot.setdefault(thr, np.zeros(5))
            r += (hit, len(gt), len(pk), tok, tn)
        print(mid, hname, "actions", len(gt), flush=True)
    print("threshold | recall precision | team accuracy at matched possession actions")
    for thr, (hit, n, npk, tok, tn) in tot.items():
        print("{:.1f} | {:.3f} {:.3f} | {:.3f} ({} actions)".format(thr, hit / max(n, 1), hit / max(npk, 1), tok / max(tn, 1), int(tn)))


if __name__ == "__main__":
    main(sys.argv[1])
