"""P9bi full chain: v18's out-of-fold chain (v16 OOF, P9at's BigGRU members, P9au's wide-TCN members) with the boundary
GRU's probability b (the switch cost 12 (1 - b) + 0.5 b) from the three-draw boundary GRU (cache/oof_bnd3); scored at
1.6 s and at 3.0 s; the single adoption rule on that swap only. Exit 0 when it holds. CPU.

    python analysis/p9bi_chain.py      # -> eda_out/reports/P9bi_chain.md
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

C = lambda *p: os.path.join(ROOT, "cache", *p)
OUT = os.path.join(ROOT, "eda_out", "reports", "P9bi_chain.md")


def main():
    rows = []
    for f in sorted(glob.glob(C("oof_bnd3", "*.npz"))):
        n = os.path.basename(f)
        z, big, t1, w = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f))
        g, lab = z["g"], z["labels"]
        assert (w["g"] == g).all()
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        l3 = merged(lab, 3.0)
        r = [n[4:-4]]
        p = (big["pg"] + t1["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6
        for bb in (z["b"], w["b"]):
            sw = HI * (1 - bb) + LO * bb
            c = np.where(g, decode(p, g & ~dead, sw), pm.X).astype(np.int8)
            r += [pm.spell_scores(lab, c, pm.valid_runs(lab))["mean_f1"], pm.spell_scores(l3, c, pm.valid_runs(l3))["mean_f1"]]
        rows.append(r)
    lines = ["# P9bi: v18 OOF zinciri, sınır GRU'su 3 çekilişle", "", "`analysis/p9bi_chain.py`.", "",
             "| maç | v18 | v18 @3s | d3 | d3 @3s |", "|---|---|---|---|---|"]
    lines += ["| {} | ".format(r[0]) + " | ".join("{:.4f}".format(x) for x in r[1:]) + " |" for r in rows]
    lines.append("")
    ok = len(rows) == 10
    for col, name in ((2, "d3"),):
        d = {}
        for src in ("DFL", "metrica"):
            q = np.array([r[1:] for r in rows if src in r[0]])
            dd = q[:, col:col + 2] - q[:, :2]
            d[src] = dd.mean(0)
            lines.append("{}: {} − v18 1.6 s {:+.4f} ({} / {}), 3 s {:+.4f} ({} / {})".format(src, name, dd[:, 0].mean(), int((dd[:, 0] > 0).sum()), len(dd), dd[:, 1].mean(), int((dd[:, 1] > 0).sum()), len(dd)))
        if name == "d3":
            ok = ok and d["DFL"][0] > 0 and d["DFL"][1] >= 0 and d["metrica"][0] >= 0 and d["metrica"][1] >= 0
    lines.append("karar: " + ("tuttu" if ok else "alınmadı / eksik"))
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
