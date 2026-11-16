"""Split P9b's "true edges" gain (+0.012) into what an edge method could reach (research report "Yapısal sıçrama
adayları", item 2): on P9a's out-of-fold chains (v14 six-net team, production switch cost, dead from the OOF dead
probabilities), predicted blocks matched to true ones are fixed one kind at a time: one-to-one blocks' starts only,
their ends only, both; merged predicted blocks (covering 2+ true blocks) split; split true blocks (covered by 2+
predicted blocks) merged. Also the offset distribution of one-to-one starts and ends. CPU.

    python analysis/p9z_dead_edge_split.py <oof_dir>      # -> eda_out/reports/P9z_dead_edge_split.md
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

FPS = 25
OUT = os.path.join(ROOT, "eda_out", "reports", "P9z_dead_edge_split.md")


def overlaps(x, ys):
    return [(c, d) for c, d in ys if d > x[0] and c < x[1]]


def fixes(pred, truth):
    """The dead masks with one kind of block error fixed from the labels, and the one-to-one offsets (frames)."""
    P, T = blocks(pred), blocks(truth)
    out = {k: pred.copy() for k in ("starts", "ends", "one-to-one both", "merges split", "splits merged", "all matched",
                                    "starts > 0.5 s", "starts <= 0.5 s", "starts early", "starts late", "ends > 0.5 s")}
    offs = []
    for a, b in P:
        ts = overlaps((a, b), T)
        if not ts:
            continue
        out["all matched"][a:b] = False
        for c, d in ts:
            out["all matched"][c:d] = True
        if len(ts) > 1:
            out["merges split"][a:b] = False
            for c, d in ts:
                out["merges split"][c:d] = True
            continue
        c, d = ts[0]
        if len(overlaps((c, d), P)) > 1:
            continue
        offs.append((a - c, b - d))
        big = abs(a - c) > FPS // 2
        todo = [("starts", (c, b)), ("ends", (a, d)), ("one-to-one both", (c, d)), ("starts > 0.5 s" if big else "starts <= 0.5 s", (c, b)),
                ("starts early" if a < c else "starts late", (c, b))]
        if abs(b - d) > FPS // 2:
            todo.append(("ends > 0.5 s", (a, d)))
        for k, (s, e) in todo:
            out[k][a:b] = False
            out[k][s:e] = True
    for c, d in T:
        ps = overlaps((c, d), P)
        if len(ps) > 1:
            out["splits merged"][ps[0][0]:ps[-1][1]] = True
    return out, offs


def main(d):
    Z = load(d)
    rows, offs = [], []
    for name in sorted(Z):
        z = Z[name]
        g, lab = z["g"], z["labels"]
        truth = (lab == pm.D) & g
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        p = (z["pg"] + z["pt"] + z["ps"] + z["mg"] + z["mt"] + z["ms"]) / 6
        cost = HI * (1 - z["b"]) + LO * z["b"]
        f = lambda dd: pm.spell_scores(lab, np.where(g, decode(p, g & ~dd, cost), pm.X).astype(np.int8), pm.valid_runs(lab))["mean_f1"]
        fx, o = fixes(dead, truth)
        offs += [(name, a, b) for a, b in o]
        r = {"match": name, "v14": f(dead)}
        r.update({k: f(v) for k, v in fx.items()})
        P, T = blocks(dead), blocks(truth)
        r["n merges"] = sum(len(overlaps(x, T)) > 1 for x in P)
        r["n splits"] = sum(len(overlaps(x, P)) > 1 for x in T)
        r["n one-to-one"] = len(o)
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    keys = ["starts", "ends", "one-to-one both", "merges split", "splits merged", "all matched", "starts > 0.5 s", "starts <= 0.5 s",
            "starts early", "starts late", "ends > 0.5 s"]
    lines = ["# P9z: ölü top kenar kazancının ayrışması (OOF, v14 tam zincir)", "", "`analysis/p9z_dead_edge_split.py`.", ""]
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        lines.append("{} (v14 {:.3f}): ".format(src, q.v14.mean()) + ", ".join("{} {:+.3f} ({} / {})".format(k, (q[k] - q.v14).mean(), int((q[k] > q.v14).sum()), len(q)) for k in keys))
        lines.append("  bloklar/maç: bire bir {:.0f}, birleşmiş {:.1f}, bölünmüş {:.1f}".format(q["n one-to-one"].mean(), q["n merges"].mean(), q["n splits"].mean()))
    od = pd.DataFrame(offs, columns=["match", "start", "end"])
    for src in ("DFL", "metrica"):
        q = od[od.match.str.startswith(src)]
        for k in ("start", "end"):
            s = q[k].abs() / FPS
            lines.append("{} {} sapması (s): medyan {:.2f}, %75 {:.2f}, %90 {:.2f}; ≤ 0.5 s %{:.0f}, ≤ 2 s %{:.0f}; tahmin erken %{:.0f}".format(
                src, k, s.median(), s.quantile(.75), s.quantile(.9), 100 * (s <= .5).mean(), 100 * (s <= 2).mean(), 100 * (q[k] < 0).mean()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
