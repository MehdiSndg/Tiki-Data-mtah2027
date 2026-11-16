"""The restart law across dead balls in the decoder (research report: 76% of DFL 0-2 s spells border a dead ball, and at
the line the restart goes to the team that did not touch the ball last), on v19's out-of-fold chain, CPU.

The production decode runs one Viterbi per alive run with a free first state, so a dead ball cuts every tie between
the team before it and the team after it. Here one Viterbi runs over each half's alive frames in order: inside a run the
change of team costs v19's per-frame cost; where a run starts after a dead block, staying with the previous run's team
costs C_SAME and changing costs nothing; in the last W frames before a dead block a change costs RHO times its usual cost
(so a short last-touch spell can open right before the stoppage). C_SAME = 0, W = 0 reproduces v19.
(C_SAME, W, RHO) chosen by nested leave one match out within each dataset; 1.6 s and 3.0 s referees; single rule.

    python analysis/p9bm_restart_decode.py      # -> eda_out/reports/P9bm_restart_decode.md
"""

import glob
import itertools
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
OUT = os.path.join(ROOT, "eda_out", "reports", "P9bm_restart_decode.md")
GRID = [(0.0, 0, 1.0)] + list(itertools.product((1.0, 2.0, 4.0, 8.0), (0, 25, 50), (1.0, 0.5, 0.25)))


def chain_decode(p_home, alive, half, sw, c_same, w, rho):
    """(n,) codes: H/A on alive frames from one two-state Viterbi per half over its alive frames; D elsewhere."""
    eps = 1e-4
    p = np.clip(np.asarray(p_home, np.float64), eps, 1 - eps)
    lh, la = np.log(p), np.log(1 - p)
    out = np.full(len(p), pm.D, np.int8)
    sw = np.asarray(sw, np.float64).copy()
    if w and rho != 1.0:
        e = np.diff(np.r_[0, alive.astype(np.int8), 0])
        for z in np.flatnonzero(e == -1):           # run ends: the next frame is dead (or the half ends)
            if z < len(p) and half[min(z, len(p) - 1)] > 0:
                sw[max(0, z - w):z] *= rho
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(alive & (half == h))
        if not len(idx):
            continue
        k = len(idx)
        start = np.r_[True, np.diff(idx) > 1]
        back = np.zeros((k, 2), np.int8)
        sh, sa = lh[idx[0]], la[idx[0]]
        for t in range(1, k):
            i = idx[t]
            if start[t]:
                hh, ah = sh - c_same, sa             # into H: from H (same team across the dead ball), from A
                aa, ha = sa - c_same, sh
            else:
                c = sw[i]
                hh, ah = sh, sa - c
                aa, ha = sa, sh - c
            back[t, 0] = 0 if hh >= ah else 1
            back[t, 1] = 1 if aa >= ha else 0
            sh = max(hh, ah) + lh[i]
            sa = max(aa, ha) + la[i]
        s = 0 if sh >= sa else 1
        path = np.empty(k, np.int8)
        for t in range(k - 1, -1, -1):
            path[t] = s
            s = back[t, s]
        out[idx] = np.where(path == 0, pm.H, pm.A)
    return out


def main():
    names, S, S0 = [], [], []
    for f in sorted(glob.glob(C("oof_v16", "*.npz"))):
        n = os.path.basename(f)
        z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (f, C("oof_big", n), C("oof_tcn128", n), C("oof_tcn128d3", n)))
        g, lab = z["g"], z["labels"]
        # OOF files carry g (played) but not the half: the halves are the played runs, in order
        e = np.diff(np.r_[0, g.astype(np.int8), 0])
        half = np.zeros(len(g), np.int8)
        for k, (a, b) in enumerate(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)), 1):
            half[a:b] = k
        runs = pm.valid_runs(lab)
        l3 = merged(lab, 3.0)
        r3 = pm.valid_runs(l3)
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        alive = g & ~dead
        b = z["b"].astype(np.float64)
        sw = HI * (1 - b) + LO * b
        p = (big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6
        sc = lambda c: (pm.spell_scores(lab, c, runs)["mean_f1"], pm.spell_scores(l3, c, r3)["mean_f1"])
        fin = lambda c: np.where(g, c, pm.X).astype(np.int8)
        S0.append(sc(fin(decode(p, alive, sw))))
        S.append([sc(fin(chain_decode(p, alive, half, sw, cs, w, r))) for cs, w, r in GRID])
        names.append(n[4:-4])
        print(names[-1], "v19 %.4f/%.4f" % S0[-1], "grid0 %.4f/%.4f" % S[-1][0], "best %.4f" % max(x[0] for x in S[-1]), flush=True)
    S, S0 = np.array(S), np.array(S0)
    lines = ["# P9bm: restart yasası ölü toplar boyunca (zincirli Viterbi), v19 OOF zinciri, iç içe LOMO", "",
             "`analysis/p9bm_restart_decode.py`. Izgara[0] = v19'un kendisi (fark {:.1e}).".format(np.abs(S[:, 0] - S0).max()), ""]
    d = {}
    for src in ("DFL", "metrica"):
        idx = [i for i, x in enumerate(names) if x.startswith(src)]
        gain, pick = [], []
        for t in idx:
            others = [i for i in idx if i != t]
            j = int(np.argmax(S[others].mean(axis=(0, 2))))
            pick.append(GRID[j])
            gain.append(S[t, j] - S0[t])
        gain = np.array(gain)
        d[src] = gain.mean(0)
        lines.append("{}: − v19 1.6 s {:+.4f} ({} / {}), 3 s {:+.4f} ({} / {}); seçilen {}".format(
            src, gain[:, 0].mean(), int((gain[:, 0] > 0).sum()), len(gain), gain[:, 1].mean(), int((gain[:, 1] > 0).sum()), len(gain), pick))
        best = S[idx].mean(axis=0)
        top = np.argsort(-best.mean(1))[:5]
        lines.append("  {} en iyi 5 (ortalama): ".format(src) + "; ".join("{} {:+.4f}/{:+.4f}".format(GRID[j], *(best[j] - S0[idx].mean(0))) for j in top))
    ok = d["DFL"][0] > 0 and d["DFL"][1] >= 0 and d["metrica"][0] >= 0 and d["metrica"][1] >= 0
    lines.append("karar: " + ("kural tuttu" if ok else "alınmadı"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
