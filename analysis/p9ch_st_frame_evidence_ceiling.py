"""P9ch: the ceiling of frame-wise possessor-team evidence (a video possessor model that says "which team has the ball"
on every frame) on fold A of SoccerTrack v2, simulated (information only; fold B unseen). Evidence comes in 1 s blocks:
a block is available with probability `cov`, and right (the label's team) with probability `acc`; it adds +-w to the
log-odds of every alive frame of the block. Compare P9cf (action-frame evidence: +0.042 perfect, +0.021 at 0.7 / 0.9).

    python analysis/p9ch_st_frame_evidence_ceiling.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from p9cf_st_evidence_forms import score  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

MIDS = ("117092", "117093", "118575", "118576")
SIMS = [(1.0, 1.0), (0.95, 1.0), (0.9, 1.0), (0.85, 1.0), (0.9, 0.5), (0.85, 0.5)]
WS = (0.25, 0.5, 1.0, 2.0)
BLOCK = 25


def main():
    rows, accs = {}, []
    for mid in MIDS:
        z = np.load(os.path.join(ROOT, "cache", "st_chain", "{}.npz".format(mid)))
        p, sw, lab, g = z["p"].astype(np.float64), z["switch"], z["labels"], z["g"]
        alive = g & np.isin(lab, (pm.H, pm.A))
        l3 = merged(lab, 3.0)
        base = score(lab, l3, p, alive, sw, g)
        accs.append(np.mean((p[alive] > 0.5) == (lab[alive] == pm.H)))
        lo = np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))
        t = np.where(lab == pm.H, 1.0, np.where(lab == pm.A, -1.0, 0.0))
        rng = np.random.default_rng(0)
        nb = len(p) // BLOCK + 1
        u = np.repeat(rng.random((nb, 2)), BLOCK, axis=0)[:len(p)]
        for acc, cov in SIMS:
            e = t * np.where(u[:, 1] <= acc, 1, -1) * (u[:, 0] <= cov)
            for w in WS:
                rows.setdefault((acc, cov, w), []).append(score(lab, l3, 1 / (1 + np.exp(-(lo + w * e))), alive, sw, g) - base)
        print(mid, "done", flush=True)
    print("chain frame team accuracy on alive frames", np.round(accs, 3))
    print("acc cov w | mean gain 1.6 s, 3 s | ahead | worst")
    for k, v in rows.items():
        v = np.array(v)
        print("{:.2f} {:.1f} {} | {:+.4f} {:+.4f} | {} / {} | {:+.4f}".format(*k, v[:, 0].mean(), v[:, 1].mean(), int((v[:, 0] > 0).sum()), len(v), v[:, 0].min()))


if __name__ == "__main__":
    main()
