"""P9a's out-of-fold chains rescored against the baseline-calibrated judge (P9k: DFL labels merged at 3.0 s fits the four
official full-match baselines best; Metrica merged the same way as a second view): does the ranking of v11 / v14 /
the multi-task heads change when the judge looks like the hidden ground truth? CPU, light.

    python analysis/p9l_rescore_calibrated.py <oof_dir>      # -> eda_out/reports/P9l_rescore_calibrated.md
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah.external import merge_short  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p9b_oof_sweeps import HI, LO, load  # noqa: E402

MERGES = (0.0, 1.6, 2.4, 3.0, 3.6)
FLATS = (4.0, 8.0, 12.0, 20.0)
OUT = os.path.join(ROOT, "eda_out", "reports", "P9l_rescore_calibrated.md")


def merged(labels, s):
    out = labels.copy()
    for a, b in pm.valid_runs(labels):
        out[a:b] = merge_short(labels[a:b], s)
    return out


def main(d):
    Z = load(d)
    rows = []
    for name, z in sorted(Z.items()):
        g = z["g"]
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        v11, mt = (z["pg"] + z["pt"] + z["ps"]) / 3, (z["mg"] + z["mt"] + z["ms"]) / 3
        cost = HI * (1 - z["b"]) + LO * z["b"]
        codes = {"v11": decode(v11, g & ~dead, cost), "mt": decode(mt, g & ~dead, cost), "v14": decode((v11 + mt) / 2, g & ~dead, cost)}
        for c in FLATS:
            codes["v14 flat {:g}".format(c)] = decode((v11 + mt) / 2, g & ~dead, c)
        for s in MERGES:
            lab = merged(z["labels"], s) if name.startswith("DFL") or s > 0 else z["labels"]
            if name.startswith("metrica"):
                lab = merged(z["labels"], s)
            r = {"match": name, "merge": s}
            for k, c in codes.items():
                r[k] = pm.spell_scores(lab, np.where(g, c, pm.X).astype(np.int8), pm.valid_runs(lab))["mean_f1"]
            rows.append(r)
        print(name, flush=True)
    df = pd.DataFrame(rows)
    keys = [k for k in df.columns if k not in ("match", "merge")]
    lines = ["# P9l: OOF zincirleri ayarlanmış hakemle (GT etiketleri birleştirilmiş)", "", "`analysis/p9l_rescore_calibrated.py`.", ""]
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        lines += ["## " + src, "", "| birleştirme | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for s, grp in q.groupby("merge"):
            lines.append("| {:g} | ".format(s) + " | ".join("{:.3f}".format(grp[k].mean()) for k in keys) + " |")
        for s, grp in q.groupby("merge"):
            lines.append("")
            lines.append("birleştirme {:g}: v14 − v11 {:+.3f} ({} / {}); en iyi düz ceza {}".format(s, (grp.v14 - grp.v11).mean(), int((grp.v14 > grp.v11).sum()), len(grp),
                                                                                      max([k for k in keys if "flat" in k], key=lambda k: grp[k].mean())))
        lines.append("")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
