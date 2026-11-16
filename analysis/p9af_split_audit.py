"""Why does our match split into fewer possessions (~261 per 90 min) than the baseline calibration expects (~286-300)?
GT-free on our match, against P9a's out-of-fold DFL / Metrica chains (v14's six-net team, production switch cost,
OOF dead ball): per alive minute, the decoded possessions, the raw probability flips (crossings of 0.4 / 0.6 with
hysteresis), the share of confident frames, the mean boundary value, dead blocks per 90 min and the decoded
possession durations; for DFL also the true possessions (labels, and merged at 3 s). Our match's p is v16's (with the
crop and restart evidence), from the build's MTAH_DUMP. No labels of our match are read. CPU.

    python analysis/p9af_split_audit.py <oof_dir> <v16_dump.npz>      # -> eda_out/reports/P9af_split_audit.md
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
from mtah.teammodel import decode  # noqa: E402
from p9b_oof_sweeps import HI, LO, blocks, load  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "eda_out", "reports", "P9af_split_audit.md")


def flips(p, alive):
    n, state = 0, None
    for v, a in zip(p, alive):
        if not a:
            state = None
            continue
        s = 1 if v > 0.6 else (0 if v < 0.4 else state)
        if state is not None and s is not None and s != state:
            n += 1
        state = s if s is not None else state
    return n


def stats(codes, p, alive, dead, g, b):
    amin = alive.sum() / FPS / 60
    gmin = g.sum() / FPS / 60
    sp = pm.possession_spells(codes, pm.valid_runs(codes))
    d = np.array([(e - s) / FPS for s, e, _ in sp])
    return {"alive share": alive.sum() / g.sum(), "poss / alive min": len(sp) / amin, "poss / 90": len(sp) / gmin * 90,
            "flips / alive min": flips(p, alive) / amin, "confident share": np.mean(np.abs(p[alive] - 0.5) > 0.4),
            "mean b": b[alive].mean(), "dead blocks / 90": len(blocks(dead & g)) / gmin * 90,
            "dur median s": np.median(d), "dur < 2 s share": np.mean(d < 2), "dur > 20 s share": np.mean(d > 20)}


def main(d, dump):
    Z = load(d)
    rows = []
    for name, z in sorted(Z.items()):
        g = z["g"]
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        p = (z["pg"] + z["pt"] + z["ps"] + z["mg"] + z["mt"] + z["ms"]) / 6
        codes = np.where(g, decode(p, g & ~dead, HI * (1 - z["b"]) + LO * z["b"]), pm.X).astype(np.int8)
        r = {"match": name}
        r.update(stats(codes, p, g & ~dead, dead, g, z["b"]))
        lab = z["labels"]
        gmin = g.sum() / FPS / 60
        r["true poss / 90"] = len(pm.possession_spells(lab, pm.valid_runs(lab))) / gmin * 90
        l3 = merged(lab, 3.0)
        r["true 3 s poss / 90"] = len(pm.possession_spells(l3, pm.valid_runs(l3))) / gmin * 90
        rows.append(r)
    u = np.load(dump)
    g, dead, p = u["g"], u["dead"] & u["g"], u["p"]
    b = (HI - u["switch"]) / (HI - LO)
    codes = np.where(g, decode(p, g & ~dead, u["switch"]), pm.X).astype(np.int8)
    r = {"match": "bizim maç (v16)"}
    r.update(stats(codes, p, g & ~dead, dead, g, b))
    rows.append(r)
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c != "match"]
    groups = [("DFL", df[df.match.str.startswith("DFL")]), ("Metrica", df[df.match.str.startswith("metrica")]), ("bizim maç", df[df.match.str.startswith("bizim")])]
    lines = ["# P9af: bizim maç neden az bölünüyor? (GT'siz, OOF ile karşılaştırma)", "", "`analysis/p9af_split_audit.py`.", "",
             "| ölçü | " + " | ".join(n for n, _ in groups) + " |", "|---|" + "---|" * len(groups)]
    for c in cols:
        lines.append("| {} | ".format(c) + " | ".join("{:.3f}".format(q[c].mean()) if len(q) and q[c].notna().any() else "–" for _, q in groups) + " |")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
