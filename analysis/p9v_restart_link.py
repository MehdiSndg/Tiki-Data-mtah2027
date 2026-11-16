"""Link the decode across dead balls with the restart law (research report "Yapısal sıçrama adayları", item 1): 76% of
0-2 s spells border a dead ball, and at a line stoppage the restart goes to the opponent of the last toucher 78% of the
time, while decode() starts every alive run afresh. On P9a's out-of-fold chains (v14's six-net team, production switch
cost, v11 dead ball), CPU:

0. diagnosis: of v14's missed true spells <= 2 s, the share next to a dead ball; how often v14's first spell after a
   true restart has the wrong team.
1. rule: for each predicted dead block, R = the team v14 decodes right after it; if the ball's last sighting before the
   block is within GATE m of a touch / goal line and the frames just before the block are decoded as R, the last LEN s
   before the block become the opponent of R (the "last touch before out" spell). LEN and GATE are chosen nested on DFL.

    python analysis/p9v_restart_link.py <oof_dir>      # -> eda_out/reports/P9v_restart_link.md
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
from p9i_edge_refine import rebuild  # noqa: E402

FPS = 25
LENS = (0.5, 1.0, 1.5)
GATES = (1.0, 2.0, 3.0, 5.0)
OUT = os.path.join(ROOT, "eda_out", "reports", "P9v_restart_link.md")


def line_dist(m):
    """Per frame, the distance (m) of the ball's last sighting to the nearest touch or goal line."""
    L, W = m["pitch"]
    x = pd.Series(np.asarray(m["bx"], float) * L).ffill().values
    y = pd.Series(np.asarray(m["by"], float) * W).ffill().values
    return np.minimum.reduce([x, L - x, y, W - y])


def snap_starts(dead, m, win=3 * FPS, lag=4):
    """Each predicted dead block's start moved to `lag` frames after the ball's last sighting inside the pitch, when the
    next sighting (or the ball itself) is outside it within +-win frames of the predicted start (DFL: 66% of blocks start
    at the ball leaving the pitch, median +0.16 s)."""
    L, W = m["pitch"]
    x, y = np.asarray(m["bx"], float) * L, np.asarray(m["by"], float) * W
    seen = np.isfinite(x)
    inside = seen & (x >= 0) & (x <= L) & (y >= 0) & (y <= W)
    out = dead.copy()
    for a, b in blocks(dead):
        lo, hi = max(1, a - win), min(len(dead) - 1, a + win)
        cand = [i for i in range(lo, hi) if inside[i] and not inside[i + 1] and (seen[i + 1] or not seen[i + 1])]
        if not cand:
            continue
        c = min(cand, key=lambda i: abs(i - a)) + lag
        if c < a:
            out[c:a] = True
        elif c < b:
            out[a:c] = False
    return out


def link(codes, dead, ld, length, gate):
    out = codes.copy()
    n = int(length * FPS)
    for a, b in blocks(dead):
        if a < n + 1 or b >= len(codes):
            continue
        r = codes[b]
        if r not in (pm.H, pm.A) or ld[a - 1] > gate:
            continue
        seg = slice(a - n, a)
        if np.all(codes[seg] == r):
            out[seg] = pm.A if r == pm.H else pm.H
    return out


def score(lab, codes, g):
    return pm.spell_scores(lab, np.where(g, codes, pm.X).astype(np.int8), pm.valid_runs(lab))["mean_f1"]


def main(d):
    Z = load(d)
    rows, diag = [], []
    for name in sorted(Z):
        z = Z[name]
        g, lab = z["g"], z["labels"]
        m = rebuild(name)
        ld = line_dist(m)
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        p = (z["pg"] + z["pt"] + z["ps"] + z["mg"] + z["mt"] + z["ms"]) / 6
        codes = decode(p, g & ~dead, HI * (1 - z["b"]) + LO * z["b"])
        codes = np.where(g, codes, pm.X).astype(np.int8)
        # diagnosis
        tsp = pm.possession_spells(lab, pm.valid_runs(lab))
        psp = pm.possession_spells(codes, pm.valid_runs(lab))
        short = [(a, b, t) for a, b, t in tsp if (b - a) / FPS <= 2]
        hit = lambda a, b, t: any(t2 == t and min(b, b2) - max(a, a2) > 0.5 * (b - a) for a2, b2, t2 in psp)
        missed = [(a, b, t) for a, b, t in short if not hit(a, b, t)]
        adj = lambda a, b: (a > 0 and lab[a - 1] == pm.D) or (b < len(lab) and lab[b] == pm.D)
        true_dead = (lab == pm.D) & g
        restarts = [(a, b) for a, b in blocks(true_dead) if b < len(lab) and lab[b] in (pm.H, pm.A)]
        wrong_first = np.mean([codes[min(len(codes) - 1, b + 12)] != lab[b] for a, b in restarts if codes[min(len(codes) - 1, b + 12)] in (pm.H, pm.A)])
        diag.append({"match": name, "short": len(short), "missed": len(missed), "missed_dead_adjacent": np.mean([adj(a, b) for a, b, _ in missed]) if missed else np.nan,
                     "restart_first_wrong": wrong_first})
        r = {"match": name, "v14": score(lab, codes, g)}
        sdead = snap_starts(dead, m) & g
        scodes = np.where(g, decode(p, g & ~sdead, HI * (1 - z["b"]) + LO * z["b"]), pm.X).astype(np.int8)
        r["snap"] = score(lab, scodes, g)
        for L_ in LENS:
            for G_ in GATES:
                r["snap+L{} G{}".format(L_, G_)] = score(lab, link(scodes, sdead, ld, L_, G_), g)
        for L_ in LENS:
            for G_ in GATES:
                r["L{} G{}".format(L_, G_)] = score(lab, link(codes, dead, ld, L_, G_), g)
        rows.append(r)
        print(diag[-1], {k: round(v, 3) for k, v in r.items() if k != "match"}, flush=True)
    df, dg = pd.DataFrame(rows), pd.DataFrame(diag)
    keys = [k for k in df.columns if k not in ("match", "v14")]
    lines = ["# P9v: ölü top boyunca restart kuralı (OOF, v14 tam zincir)", "", "`analysis/p9v_restart_link.py`.", "", "## Teşhis", "",
             "| maç | ≤ 2 s spell | kaçan | kaçanlarda ölüye komşu | restart sonrası ilk takım yanlış |", "|---|---|---|---|---|"]
    lines += ["| {} | {} | {} | {:.2f} | {:.2f} |".format(r.match, r.short, r.missed, r.missed_dead_adjacent, r.restart_first_wrong) for _, r in dg.iterrows()]
    lines += ["", "## Kural", ""]
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        best = sorted(keys, key=lambda k: -(q[k] - q.v14).mean())[:5]
        lines.append("{}: v14 {:.3f}; en iyi: ".format(src, q.v14.mean()) + ", ".join("{} {:+.3f} ({} / {})".format(k, (q[k] - q.v14).mean(), int((q[k] > q.v14).sum()), len(q)) for k in best))
    dfl = df[df.match.str.startswith("DFL")]
    nest = []
    for _, r in dfl.iterrows():
        k = max(keys, key=lambda k: (dfl[dfl.match != r.match][k] - dfl[dfl.match != r.match].v14).mean())
        nest.append(r[k] - r.v14)
    met = df[df.match.str.startswith("metrica")]
    kall = max(keys, key=lambda k: (dfl[k] - dfl.v14).mean())
    lines += ["", "İç içe (DFL): {:+.3f}, {} / {}; DFL'de seçilen {} Metrica'da {:+.3f} ({} / {}).".format(
        np.mean(nest), int(np.sum(np.array(nest) > 0)), len(nest), kall, (met[kall] - met.v14).mean(), int((met[kall] > met.v14).sum()), len(met))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
