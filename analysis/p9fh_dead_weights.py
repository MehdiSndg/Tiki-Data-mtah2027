"""P9fh: weights of the dead-ball probabilities on SoccerTrack's full chain (real tracking + cross-fold video).
The oracle on SoccerTrack (16 Nov): perfect dead ball 0.7235 -> 0.7959 (+0.072), three times the +0.025 measured on
DFL copies in October. Production: P(dead) = (classifier + GRU + video) / 3 (+ Mali's on our match). Here
P(dead) = w * video + (1 - w) * (classifier + GRU) / 2 for w in a grid, P9di's inputs (cache/p9di), leave-one-match-out pick.

    .venv/bin/python analysis/p9fh_dead_weights.py

Registered before the run (16 Nov): a new weight is adopted only if the nested pick beats w = 1/3 by >= +0.003 at both
referees and on >= 6 of 8 matches, and the label-free agreement on our match does not argue against it (reported).
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402

WS = (0.0, 1 / 6, 1 / 3, 0.5, 2 / 3, 5 / 6, 1.0)


def main():
    from mtah import semimarkov as sm
    from p9l_rescore_calibrated import merged
    S, names = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "cache", "p9di", "in_*.npz"))):
        z = np.load(f)
        g, lab = z["g"], z["lab"]
        l3 = merged(lab, 3.0)
        d2, d3 = z["dead2"].astype(np.float64), z["dead3"].astype(np.float64)
        v = np.clip(3 * d3 - 2 * d2, 0, 1)
        row = []
        for w in WS:
            c = np.where(g, sm.decode_halves(z["team"].astype(np.float64), w * v + (1 - w) * d2, g, z["b"].astype(np.float64)), pm.X).astype(np.int8)
            row.append([pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)])
        S.append(row)
        names.append(os.path.basename(f)[3:-4])
        print(names[-1], [round(r[0], 4) for r in row], flush=True)
    S = np.array(S)
    base = WS.index(1 / 3)
    print("w:", [round(w, 3) for w in WS])
    print("mean 1.6 s:", np.round(S[:, :, 0].mean(0), 4).tolist())
    print("mean 3 s:  ", np.round(S[:, :, 1].mean(0), 4).tolist())
    nested, picks = [], []
    for i in range(len(names)):
        keep = [j for j in range(len(names)) if j != i]
        k = int(np.argmax(S[keep].mean(axis=(0, 2))))
        picks.append(round(WS[k], 3))
        nested.append(S[i, k])
    d = np.array(nested) - S[:, base]
    ok = (d.mean(0) >= 0.003).all() and int((d[:, 0] > 0).sum()) >= 6
    print("fold picks", picks, "nested diff {:+.4f} / {:+.4f} ({}/8 ahead) -> {}".format(*d.mean(0), int((d[:, 0] > 0).sum()), "holds" if ok else "FAILS"))


if __name__ == "__main__":
    main()
