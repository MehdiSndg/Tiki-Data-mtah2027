"""P9cj: P9cf's pair form with false alarms (fold A of SoccerTrack v2 only, simulated evidence): besides recall, team
accuracy and +-3 frames jitter, detections that are no action at all, at uniformly random frames with a random team, so
that the precision of the detections is `prec`. Fixes the decision threshold for the T-DEED test before its result.

    python analysis/p9cj_st_evidence_false_alarms.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah.external import from_soccertrack  # noqa: E402
from p9cf_st_evidence_forms import actions, fuse, score  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

MIDS = ("117092", "117093", "118575", "118576")
SIMS = [(a, r, pr) for a, r in ((0.9, 0.7), (0.9, 0.6), (0.85, 0.7), (0.95, 0.8)) for pr in (1.0, 0.85, 0.7, 0.55)]
WS = (0.5, 1.0, 2.0)


def main():
    rows = {}
    for mid in MIDS:
        z = np.load(os.path.join(ROOT, "cache", "st_chain", "{}.npz".format(mid)))
        p, sw, lab, g = z["p"].astype(np.float64), z["switch"], z["labels"], z["g"]
        acts = actions(mid, from_soccertrack(mid)["half"])
        alive = g & np.isin(lab, (pm.H, pm.A))
        l3 = merged(lab, 3.0)
        base = score(lab, l3, p, alive, sw, g)
        lo = np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))
        rng = np.random.default_rng(0)
        u, j = rng.random((len(acts), 2)), rng.integers(-3, 4, len(acts))
        live = np.flatnonzero(alive)
        fk, ft = rng.choice(live, len(acts)), rng.choice((-1, 1), len(acts))
        for acc, rec, prec in SIMS:
            ev = [(k + j[i], t if u[i, 1] <= acc else -t) for i, (k, t) in enumerate(acts) if u[i, 0] <= rec]
            nf = int(round(len(ev) * (1 - prec) / prec))
            ev = sorted(ev + [(int(fk[i]), int(ft[i])) for i in range(nf)])
            for w in WS:
                q, s = fuse("pair", w, ev, lo, sw)
                rows.setdefault((acc, rec, prec, w), []).append(score(lab, l3, q, alive, s, g) - base)
        print(mid, "done", flush=True)
    print("acc recall precision w | mean gain 1.6 s, 3 s | ahead | worst")
    for k, v in rows.items():
        v = np.array(v)
        print("{:.2f} {:.1f} {:.2f} {} | {:+.4f} {:+.4f} | {} / {} | {:+.4f}".format(*k, v[:, 0].mean(), v[:, 1].mean(), int((v[:, 0] > 0).sum()), len(v), v[:, 0].min()))


if __name__ == "__main__":
    main()
