"""P9au full chain: v17's out-of-fold chain (v16 OOF with P9at's BigGRU members) with the TCN members (single head pt and
two-head mt) replaced by the wide TCN (dilations 1-128, cache/oof_tcn128), and with the two averaged; scored at 1.6 s and
against labels merged at 3.0 s; the single adoption rule (DFL and Metrica not behind at either referee, DFL ahead at
1.6 s) on the swap, the pre-planned variant (P9ao). CPU.

    python analysis/p9au_chain.py      # -> eda_out/reports/P9au_chain.md
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
OUT = os.path.join(ROOT, "eda_out", "reports", "P9au_chain.md")


def main():
    rows = []
    for f in sorted(glob.glob(C("oof_tcn128", "*.npz"))):
        n = os.path.basename(f)
        z, big, w = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), f))
        g, lab = z["g"], z["labels"]
        assert (w["g"] == g).all()
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        l3 = merged(lab, 3.0)
        sw = HI * (1 - z["b"]) + LO * z["b"]
        r = [n[4:-4]]
        for pt, mt in ((z["pt"], z["mt"]), (w["pt"], w["mt"]), ((z["pt"] + w["pt"]) / 2, (z["mt"] + w["mt"]) / 2)):
            p = (big["pg"] + pt + z["ps"] + big["mg"] + mt + z["ms"]) / 6
            c = np.where(g, decode(p, g & ~dead, sw), pm.X).astype(np.int8)
            r += [pm.spell_scores(lab, c, pm.valid_runs(lab))["mean_f1"], pm.spell_scores(l3, c, pm.valid_runs(l3))["mean_f1"]]
        rows.append(r)
    lines = ["# P9au: v17 OOF zinciri, TCN üyeleri yerine geniş TCN (1–128) / ikisinin ortalaması", "", "`analysis/p9au_chain.py`.", "",
             "| maç | v17 | v17 @3s | wide | wide @3s | avg | avg @3s |", "|---|---|---|---|---|---|---|"]
    lines += ["| {} | ".format(r[0]) + " | ".join("{:.4f}".format(x) for x in r[1:]) + " |" for r in rows]
    lines.append("")
    ok = len(rows) == 10
    for col, name in ((2, "wide"), (4, "avg")):
        d = {}
        for src in ("DFL", "metrica"):
            q = np.array([r[1:] for r in rows if src in r[0]])
            dd = q[:, col:col + 2] - q[:, :2]
            d[src] = dd.mean(0)
            lines.append("{}: {} − v17 1.6 s {:+.4f} ({} / {}), 3 s {:+.4f} ({} / {})".format(src, name, dd[:, 0].mean(), int((dd[:, 0] > 0).sum()), len(dd), dd[:, 1].mean(), int((dd[:, 1] > 0).sum()), len(dd)))
        if name == "wide":
            ok = ok and d["DFL"][0] > 0 and d["DFL"][1] >= 0 and d["metrica"][0] >= 0 and d["metrica"][1] >= 0
    lines.append("karar (yalnız 'wide', önceden planlanan): " + ("tuttu" if ok else "alınmadı") + "; 'avg' bilgi")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
