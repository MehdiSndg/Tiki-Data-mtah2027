"""P9ce: what action-frame team evidence can add to v19c's team call on SoccerTrack v2 (information, not a selection:
no parameter is carried from here; the fusion weight stays to be tuned on the training fold only, as registered).

Before GPU hours go into the video models: the chain of P9bw (cache/st_chain: p, switch, labels) with simulated
evidence "team T acts on the ball at frame f" at the annotated ball actions. Each action is detected with probability
`recall`, its team is right with probability `acc`, its time is jittered by +-`jit` frames (uniform); the evidence adds
+-W to the log-odds of home over the frames f-HW..f+HW (log-linear stream weight, the fusion form of the literature
review). Dead ball from the labels, as P9bw. acc 1 / recall 1 is the ceiling of any action-frame video evidence; the
status labels are event-anchored there, so it is an optimistic ceiling.

    python analysis/p9ce_st_evidence_ceiling.py
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
from mtah.teammodel import decode  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

MIDS = ("117092", "117093", "118575", "118576", "118577", "118578", "128057", "128058")
HW = 6
GRID = [(1.0, 1.0, 0), (0.9, 1.0, 0), (0.9, 0.7, 3), (0.8, 0.7, 3), (0.75, 0.6, 3), (0.65, 0.6, 3)]
WS = (2.0, 4.0, 8.0)


def actions(mid, half):
    """[(frame index in the concatenated match, 1 home / 0 away)] of the actions with a team."""
    acts = json.load(open(os.path.join(SOCCERTRACK, "bas", mid, "{}_12_class_events.json".format(mid))))["actions"]
    n1 = int(np.sum(half == 1))
    out = []
    for h, hname in ((1, "1st"), (2, "2nd")):
        st = np.load(os.path.join(SOCCERTRACK, "ball", "{}_{}_ball.npz".format(mid, hname)))["status"]
        ev = [a for a in acts if a["gameTime"].strip().startswith("{} ".format(h))]
        codes = soccertrack_side_codes(ev, st)
        for a in ev:
            if codes.get(a.get("team")) is None:
                continue
            k = int(a["frame"]) + (0 if h == 1 else n1 + 1)
            if k < len(half) and half[k] == h:
                out.append((k, 1 if codes[a["team"]] == 1 else 0))
    return out


def score(lab, l3, p, alive, sw, g):
    c = np.where(g, np.where(alive, decode(p, alive, sw), pm.D), pm.X).astype(np.int8)
    return (pm.spell_scores(lab, c, pm.valid_runs(lab))["mean_f1"], pm.spell_scores(l3, c, pm.valid_runs(l3))["mean_f1"])


def main():
    rows = {}
    for mid in MIDS:
        z = np.load(os.path.join(ROOT, "cache", "st_chain", "{}.npz".format(mid)))
        p, sw, lab, g = z["p"].astype(np.float64), z["switch"], z["labels"], z["g"]
        half = from_soccertrack(mid)["half"]
        acts = actions(mid, half)
        alive = g & np.isin(lab, (pm.H, pm.A))
        l3 = merged(lab, 3.0)
        base = score(lab, l3, p, alive, sw, g)
        lo = np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))
        rng = np.random.default_rng(0)
        u = rng.random((len(acts), 2))
        j = rng.integers(-3, 4, len(acts))
        print(mid, "actions", len(acts), "base {:.4f} {:.4f}".format(*base), flush=True)
        for acc, rec, jit in GRID:
            for W in WS:
                e = np.zeros(len(p))
                for i, (k, t) in enumerate(acts):
                    if u[i, 0] > rec:
                        continue
                    s = (1 if t else -1) * (1 if u[i, 1] <= acc else -1)
                    c = k + (j[i] if jit else 0)
                    e[max(c - HW, 0):c + HW + 1] += s * W
                q = 1 / (1 + np.exp(-(lo + np.clip(e, -2 * W, 2 * W))))
                r = score(lab, l3, q, alive, sw, g)
                rows.setdefault((acc, rec, jit, W), []).append((r[0] - base[0], r[1] - base[1]))
    print("acc recall jitter W | mean gain 1.6 s, 3 s | matches ahead at 1.6 s | worst match")
    for k, v in rows.items():
        v = np.array(v)
        print("{:.2f} {:.1f} {} {:.0f} | {:+.4f} {:+.4f} | {} / {} | {:+.4f}".format(*k, v[:, 0].mean(), v[:, 1].mean(),
                                                                                int((v[:, 0] > 0).sum()), len(v), v[:, 0].min()))


if __name__ == "__main__":
    main()
