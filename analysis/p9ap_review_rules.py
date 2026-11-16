"""Two decode-side ideas from an external review, on P9a's out-of-fold chains (v14's six-net team, its boundary GRU,
the OOF dead ball), CPU:

A. switch cost lowered where the team probability itself jumps: cost = HI (1 - b') + LO b', b' = max(b, k * |p[t + w] -
   p[t - w]|) for w in WS frames and k in KS.
B. dead-ball guard by team: an alive stretch of the OOF dead probability between LO_D and HI_D, flanked by the same
   decoded team, opened as D when the ball was last seen within 3 m of a line ("open"); a weak predicted dead block
   (mean probability below VETO) flanked by the same team with the ball seen inside the pitch and moving, closed
   ("veto").
Nested choice on DFL (each match decoded with the setting best on the other six), checked on Metrica; also against
labels merged at 3.0 s.

    python analysis/p9ap_review_rules.py <oof_dir>      # -> eda_out/reports/P9ap_review_rules.md
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
from p9l_rescore_calibrated import merged  # noqa: E402
from p9v_restart_link import line_dist  # noqa: E402

FPS = 25
WS, KS = (5, 12, 25), (0.5, 1.0, 2.0)
OPEN = ((0.30, 0.50), (0.35, 0.50), (0.40, 0.50))
VETO = (0.55, 0.60, 0.65)
OUT = os.path.join(ROOT, "eda_out", "reports", "P9ap_review_rules.md")


def jump(p, w):
    q = np.pad(p, w, mode="edge")
    return np.abs(q[2 * w:] - q[:-2 * w])


def chain(p, dead, g, cost):
    return np.where(g, decode(p, g & ~dead, cost), pm.X).astype(np.int8)


def guard(codes, q, dead, g, ld, m, open_band=None, veto=None):
    out = dead.copy()
    L, W = m["pitch"]
    if open_band is not None:
        lo, hi = open_band
        cand = g & ~dead & (q >= lo) & (q < hi)
        for a, b in blocks(cand):
            if b - a < FPS or a == 0 or b >= len(codes):
                continue
            if codes[a - 1] == codes[b] and codes[a - 1] in (pm.H, pm.A) and ld[a] <= 3.0:
                out[a:b] = True
    if veto is not None:
        x, y = np.asarray(m["bx"], float) * L, np.asarray(m["by"], float) * W
        v = np.r_[0, np.hypot(np.diff(x), np.diff(y)) * FPS]
        for a, b in blocks(dead & g):
            if a == 0 or b >= len(codes) or q[a:b].mean() >= veto:
                continue
            seen = np.isfinite(x[a:b])
            inside = seen & (x[a:b] > 1) & (x[a:b] < L - 1) & (y[a:b] > 1) & (y[a:b] < W - 1)
            if codes[a - 1] == codes[b] and codes[a - 1] in (pm.H, pm.A) and inside.mean() > 0.8 and np.nanmedian(v[a:b]) > 2.0:
                out[a:b] = False
    return out


def main(d):
    Z = load(d)
    rows = []
    for name, z in sorted(Z.items()):
        g, lab = z["g"], z["labels"]
        m = rebuild(name)
        ld = line_dist(m)
        q = (z["qh"] + z["qg"]) / 2
        dead = to_blocks(q) & g
        p = (z["pg"] + z["pt"] + z["ps"] + z["mg"] + z["mt"] + z["ms"]) / 6
        base_cost = HI * (1 - z["b"]) + LO * z["b"]
        l3 = merged(lab, 3.0)
        sc = lambda c: (pm.spell_scores(lab, c, pm.valid_runs(lab))["mean_f1"], pm.spell_scores(l3, c, pm.valid_runs(l3))["mean_f1"])
        codes = chain(p, dead, g, base_cost)
        r = {"match": name}
        r["base"], r["base@3"] = sc(codes)
        for w in WS:
            for k in KS:
                bb = np.maximum(z["b"], np.clip(k * jump(p, w), 0, 1))
                r["A w{} k{}".format(w, k)], r["A w{} k{} @3".format(w, k)] = sc(chain(p, dead, g, HI * (1 - bb) + LO * bb))
        for ob in OPEN:
            dd = guard(codes, q, dead, g, ld, m, open_band=ob)
            r["B open {}-{}".format(*ob)], r["B open {}-{} @3".format(*ob)] = sc(chain(p, dd, g, base_cost))
        for vt in VETO:
            dd = guard(codes, q, dead, g, ld, m, veto=vt)
            r["B veto {}".format(vt)], r["B veto {} @3".format(vt)] = sc(chain(p, dd, g, base_cost))
        rows.append(r)
        print(name, flush=True)
    df = pd.DataFrame(rows)
    keys = [k for k in df.columns if k not in ("match", "base", "base@3") and not k.endswith("@3")]
    dfl, met = df[df.match.str.startswith("DFL")], df[df.match.str.startswith("metrica")]
    lines = ["# P9ap: dış incelemenin iki kod çözme fikri (OOF v14 tam zincir)", "", "`analysis/p9ap_review_rules.py`.", ""]
    for k in keys:
        lines.append("- {}: DFL {:+.4f} ({} / 7), @3 {:+.4f}; Metrica {:+.4f} ({} / 3), @3 {:+.4f}".format(
            k, (dfl[k] - dfl.base).mean(), int((dfl[k] > dfl.base).sum()), (dfl[k + " @3"] - dfl["base@3"]).mean(),
            (met[k] - met.base).mean(), int((met[k] > met.base).sum()), (met[k + " @3"] - met["base@3"]).mean()))
    for fam in ("A ", "B open", "B veto"):
        ks = [k for k in keys if k.startswith(fam)]
        nest = [r[max(ks, key=lambda k: (dfl[dfl.match != r.match][k] - dfl[dfl.match != r.match].base).mean())] - r.base for _, r in dfl.iterrows()]
        lines.append("{} iç içe DFL: {:+.4f}, {} / 7".format(fam.strip(), np.mean(nest), int(np.sum(np.array(nest) > 0))))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
