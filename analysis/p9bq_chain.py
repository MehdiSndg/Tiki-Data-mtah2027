"""P9bq full chain: v19's out-of-fold chain (v16 OOF, P9at's BigGRU members, the two-head wide TCN of P9au) with the
three-draw single-head wide TCN (cache/oof_tcn128d3, v19's member) replaced by the five-draw one (cache/oof_tcn128d5);
scored at 1.6 s and 3.0 s; the single adoption rule (DFL and Metrica not behind at either referee, DFL ahead at 1.6 s),
written down before the run (9 Nov). Exit 0 when it holds. CPU.

    python analysis/p9bq_chain.py      # -> eda_out/reports/P9bq_chain.md
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
OUT = os.path.join(ROOT, "eda_out", "reports", "P9bq_chain.md")


def main():
    rows = []
    for f in sorted(glob.glob(C("oof_tcn128d5", "*.npz"))):
        n = os.path.basename(f)
        z, big, t1, d3, w = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), C("oof_tcn128d3", n), f))
        g, lab = z["g"], z["labels"]
        assert (w["g"] == g).all()
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        l3 = merged(lab, 3.0)
        sw = HI * (1 - z["b"]) + LO * z["b"]
        r = [n[4:-4]]
        for pt, mt in ((d3["pt"], t1["mt"]), (w["pt"], t1["mt"])):
            p = (big["pg"] + pt + z["ps"] + big["mg"] + mt + z["ms"]) / 6
            c = np.where(g, decode(p, g & ~dead, sw), pm.X).astype(np.int8)
            r += [pm.spell_scores(lab, c, pm.valid_runs(lab))["mean_f1"], pm.spell_scores(l3, c, pm.valid_runs(l3))["mean_f1"]]
        rows.append(r)
    lines = ["# P9bq: v19 OOF zinciri, 3 çekilişli tek başlı geniş TCN yerine 5 çekilişli olanı", "", "`analysis/p9bq_chain.py`.", "",
             "| maç | v19 | v19 @3s | d5 | d5 @3s |", "|---|---|---|---|---|"]
    lines += ["| {} | ".format(r[0]) + " | ".join("{:.4f}".format(x) for x in r[1:]) + " |" for r in rows]
    lines.append("")
    ok = len(rows) == 10
    for col, name in ((2, "d5"),):
        d = {}
        for src in ("DFL", "metrica"):
            q = np.array([r[1:] for r in rows if src in r[0]])
            if not len(q):
                d[src] = np.zeros(2) - 1
                continue
            dd = q[:, col:col + 2] - q[:, :2]
            d[src] = dd.mean(0)
            lines.append("{}: {} − v19 1.6 s {:+.4f} ({} / {}), 3 s {:+.4f} ({} / {})".format(src, name, dd[:, 0].mean(), int((dd[:, 0] > 0).sum()), len(dd), dd[:, 1].mean(), int((dd[:, 1] > 0).sum()), len(dd)))
        if name == "d5":
            ok = ok and d["DFL"][0] > 0 and d["DFL"][1] >= 0 and d["metrica"][0] >= 0 and d["metrica"][1] >= 0
    lines.append("karar: " + ("tuttu" if ok else "alınmadı / eksik"))
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
