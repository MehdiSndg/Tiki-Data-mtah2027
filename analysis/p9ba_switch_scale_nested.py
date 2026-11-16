"""Switch-cost scale k on v17's out-of-fold chain (v16 OOF with P9at's BigGRU members; cost k * (HI (1 - b) + LO b)),
including k < 1 (our match under-splits), chosen by nested leave-one-match-out within each dataset: for every held-out
match, k maximises the mean of the other matches' 1.6 s and 3 s scores. CPU.

    python analysis/p9ba_switch_scale_nested.py      # -> eda_out/reports/P9ba_switch_scale_nested.md
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p9b_oof_sweeps import HI, LO  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

K = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25)
OUT = os.path.join(ROOT, "eda_out", "reports", "P9ba_switch_scale_nested.md")


def main():
    names, S = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "cache", "oof_v16", "*.npz"))):
        n = os.path.basename(f)
        z, big = dict(np.load(f, allow_pickle=True)), dict(np.load(os.path.join(ROOT, "cache", "oof_big", n), allow_pickle=True))
        g, lab = z["g"], z["labels"]
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        l3 = merged(lab, 3.0)
        p = (big["pg"] + z["pt"] + z["ps"] + big["mg"] + z["mt"] + z["ms"]) / 6
        sw = HI * (1 - z["b"]) + LO * z["b"]
        row = []
        for k in K:
            c = np.where(g, decode(p, g & ~dead, k * sw), pm.X).astype(np.int8)
            row.append((pm.spell_scores(lab, c, pm.valid_runs(lab))["mean_f1"], pm.spell_scores(l3, c, pm.valid_runs(l3))["mean_f1"]))
        names.append(n[4:-4])
        S.append(row)
        print(n, " ".join("k%.2f %.3f/%.3f" % (k, a, b) for k, (a, b) in zip(K, row)), flush=True)
    S = np.array(S)                                   # match x k x referee
    lines = ["# P9ba: geçiş cezası ölçeği, v17 OOF zinciri, iç içe LOMO", "", "`analysis/p9ba_switch_scale_nested.py`.", ""]
    base = K.index(1.0)
    for src in ("DFL", "metrica"):
        idx = [i for i, n in enumerate(names) if n.startswith(src)]
        q = S[idx]
        lines.append("{}: ".format(src) + " ".join("k{:.2f} {:+.4f}/{:+.4f}".format(k, *(q[:, j] - q[:, base]).mean(0)) for j, k in enumerate(K)))
        pick, gain = [], []
        for t in range(len(idx)):
            others = [i for i in range(len(idx)) if i != t]
            j = int(np.argmax(q[others].mean(axis=(0, 2))))
            pick.append(K[j])
            gain.append(q[t, j] - q[t, base])
        gain = np.array(gain)
        lines.append("  iç içe seçim k = {}; seçilen − k1: 1.6 s {:+.4f} ({} / {}), 3 s {:+.4f} ({} / {})".format(
            pick, gain[:, 0].mean(), int((gain[:, 0] > 0).sum()), len(gain), gain[:, 1].mean(), int((gain[:, 1] > 0).sum()), len(gain)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
