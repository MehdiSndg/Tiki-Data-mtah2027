"""Boundary cut-and-vote on v19's out-of-fold chain (ASRF's post-processing: cut at boundary peaks, each span takes the
team its frames vote for; the literature's one segment method that helps under-splitting), two variants, CPU:

  A  cut-and-vote: within each predicted alive run, cut at local maxima of the boundary GRU's b above THR at least MIND
     frames apart; each span is H when the mean ensemble probability of home exceeds 0.5; same-team neighbours merge.
  B  insert: v19's Viterbi decode, and inside each of its spells every span between two consecutive peaks (as in A) that
     is at most MAXLEN frames long and whose mean probability votes the other team by margin MARG is flipped -- aimed at
     the 0-2 s spells the Viterbi keeps inside a long spell.

(THR, MIND) for A and (THR, MAXLEN, MARG) for B are chosen by nested leave one match out within each dataset (mean of the
1.6 s and 3 s scores of the other matches); the held-out match is scored at 1.6 s and against labels merged at 3.0 s;
the single adoption rule decides. Also the 0-2 s recall at IoU 0.5.

    python analysis/p9bl_cut_vote.py      # -> eda_out/reports/P9bl_cut_vote.md
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
OUT = os.path.join(ROOT, "eda_out", "reports", "P9bl_cut_vote.md")
GRID_A = list(itertools.product((0.2, 0.3, 0.4, 0.5, 0.6), (12, 25, 50)))
GRID_B = list(itertools.product((0.3, 0.4, 0.5, 0.6), (25, 50, 75), (0.1, 0.2, 0.3)))


def peaks(b, alive, thr, mind):
    """Frames that are local maxima of b above thr, greedily at least mind apart (highest first), inside alive frames."""
    c = np.flatnonzero(alive[1:-1] & (b[1:-1] >= thr) & (b[1:-1] >= b[:-2]) & (b[1:-1] >= b[2:])) + 1
    keep, taken = [], np.zeros(len(b), bool)
    for i in c[np.argsort(-b[c], kind="stable")]:
        if not taken[max(0, i - mind):i + mind + 1].any():
            keep.append(i)
            taken[i] = True
    return np.sort(np.array(keep, int))


def runs_of(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def cut_vote(p, b, alive, thr, mind):
    out = np.full(len(p), pm.D, np.int8)
    pk = set(peaks(b, alive, thr, mind).tolist())
    for a, z in runs_of(alive):
        cuts = [a] + [i for i in range(a + 1, z) if i in pk] + [z]
        for s, e in zip(cuts[:-1], cuts[1:]):
            out[s:e] = pm.H if p[s:e].mean() > 0.5 else pm.A
    return out


def insert(base, p, b, alive, thr, maxlen, marg):
    out = base.copy()
    pk = peaks(b, alive, thr, 1)
    for s, e in zip(pk[:-1], pk[1:]):
        if e - s > maxlen or not alive[s:e].all():
            continue
        cur = out[s:e]
        if not (cur == cur[0]).all() or cur[0] not in (pm.H, pm.A):
            continue
        m = p[s:e].mean()
        if cur[0] == pm.H and m < 0.5 - marg:
            out[s:e] = pm.A
        elif cur[0] == pm.A and m > 0.5 + marg:
            out[s:e] = pm.H
    return out


def short_recall(lab, c, runs):
    gs, ps = pm.possession_spells(lab, runs), pm.possession_spells(c, runs)
    fl = np.asarray(pm.matched_gt_flags(pm.spell_overlaps(gs, ps), len(gs), 0.5))
    d = np.array([(e - s) for s, e, _ in gs])
    return fl[d < 50].mean() if (d < 50).any() else np.nan


def main():
    names, S_base, S_A, S_B, R = [], [], [], [], []
    for f in sorted(glob.glob(C("oof_v16", "*.npz"))):
        n = os.path.basename(f)
        z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (f, C("oof_big", n), C("oof_tcn128", n), C("oof_tcn128d3", n)))
        g, lab = z["g"], z["labels"]
        runs = pm.valid_runs(lab)
        l3 = merged(lab, 3.0)
        r3 = pm.valid_runs(l3)
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        alive = g & ~dead
        b = z["b"].astype(np.float64)
        p = (big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6
        sc = lambda c: (pm.spell_scores(lab, c, runs)["mean_f1"], pm.spell_scores(l3, c, r3)["mean_f1"])
        fin = lambda c: np.where(g, c, pm.X).astype(np.int8)
        base = fin(decode(p, alive, HI * (1 - b) + LO * b))
        names.append(n[4:-4])
        S_base.append(sc(base))
        S_A.append([sc(fin(cut_vote(p, b, alive, t, m))) for t, m in GRID_A])
        S_B.append([sc(insert(base, p, b, alive, t, L, mg)) for t, L, mg in GRID_B])
        R.append(short_recall(lab, base, runs))
        print(names[-1], "base %.4f/%.4f" % S_base[-1], flush=True)
    S_base, S_A, S_B = np.array(S_base), np.array(S_A), np.array(S_B)
    lines = ["# P9bl: sınırda kes ve oyla (ASRF) / Viterbi içine kısa spell ekleme, v19 OOF zinciri, iç içe LOMO", "",
             "`analysis/p9bl_cut_vote.py`.", ""]
    verdict = {}
    for name, S, grid in (("A cut-vote", S_A, GRID_A), ("B insert", S_B, GRID_B)):
        d = {}
        for src in ("DFL", "metrica"):
            idx = [i for i, x in enumerate(names) if x.startswith(src)]
            gain, pick = [], []
            for t in idx:
                others = [i for i in idx if i != t]
                j = int(np.argmax(S[others].mean(axis=(0, 2))))
                pick.append(grid[j])
                gain.append(S[t, j] - S_base[t])
            gain = np.array(gain)
            d[src] = gain.mean(0)
            lines.append("{} {}: − v19 1.6 s {:+.4f} ({} / {}), 3 s {:+.4f} ({} / {}); seçilen {}".format(
                name, src, gain[:, 0].mean(), int((gain[:, 0] > 0).sum()), len(gain), gain[:, 1].mean(), int((gain[:, 1] > 0).sum()), len(gain), pick))
        verdict[name] = d["DFL"][0] > 0 and d["DFL"][1] >= 0 and d["metrica"][0] >= 0 and d["metrica"][1] >= 0
    lines.append("")
    lines.append("v19 0–2 s yakalama (IoU 0.5) maç ortalaması {:.3f}".format(np.nanmean(R)))
    lines.append("karar: " + ", ".join("{} {}".format(k, "tuttu" if v else "alınmadı") for k, v in verdict.items()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
