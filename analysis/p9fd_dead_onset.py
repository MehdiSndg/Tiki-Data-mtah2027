"""P9fd: the dead-ball onset is late. On SoccerTrack's full chain (P9di's decoded codes, 8 matches, 613 dead blocks) the
decoded D block starts a median 0.84 s after the label's (every match +0.64 to +0.96 s) while its end is exact (median
0.00 s); the tracking keeps a moving ball after it crosses the line, and two of the four dead-ball probabilities lag
(our match, label-free: the classifier 12 frames behind the video's dead call, Mali's 8).
shift_onsets(codes, g, k): every decoded D block that follows a possession starts k frames earlier, never eating more
than half of that possession (so no possession disappears) and never crossing a half's start.

    .venv/bin/python analysis/p9fd_dead_onset.py p9di       # quick look on P9di's codes (older chain)
    .venv/bin/python analysis/p9fd_dead_onset.py st         # v30's chain (cache/p9fc), leave-one-match-out pick of k

Registered before the st run (16 Nov): adopted only if the leave-one-match-out pick beats k = 0 at both referees by
>= +0.003 and on >= 6 of 8 matches at 1.6 s; then checked on PFF (team call with the dead ball from our dead-ball
models is not available there, so PFF is not a judge for this) and on our match label-free (share of frames changed).
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402

KS = (0, 5, 10, 15, 20, 25, 30, 40)


def shift_onsets(codes, g, k):
    out = codes.copy()
    if k <= 0:
        return out
    e = np.diff(np.r_[0, np.asarray(g).astype(np.int8), 0])
    for a0, b0 in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        segs = pm.segments(codes, [(a0, b0)])
        for i in range(1, len(segs)):
            a, b, c = segs[i]
            pa, pb, pc = segs[i - 1]
            if c != pm.D or pc not in (pm.H, pm.A):
                continue
            s = min(k, (pb - pa) // 2)
            out[a - s:a] = pm.D
    return out


def score(lab, l3, codes):
    return [pm.spell_scores(L, codes, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]


def table(rows, names):
    S = np.array([[rows[n][k] for k in KS] for n in names])      # match x k x referee
    print("k (frames):", KS)
    print("mean 1.6 s:", np.round(S[:, :, 0].mean(0), 4).tolist())
    print("mean 3 s:  ", np.round(S[:, :, 1].mean(0), 4).tolist())
    picks, nested = [], []
    for i in range(len(names)):
        keep = [j for j in range(len(names)) if j != i]
        j = int(np.argmax(S[keep].mean(axis=(0, 2))))
        picks.append(KS[j])
        nested.append(S[i, j])
    d = np.array(nested) - S[:, 0]
    ok = (d.mean(0) >= 0.003).all() and int((d[:, 0] > 0).sum()) >= 6
    print("fold picks", picks, "nested diff {:+.4f} / {:+.4f} ({}/{} ahead) -> {}".format(
        *d.mean(0), int((d[:, 0] > 0).sum()), len(names), "holds" if ok else "FAILS"))


def p9di():
    from p9l_rescore_calibrated import merged
    rows, names = {}, []
    for f in sorted(glob.glob(os.path.join(ROOT, "cache", "p9di", "out_*.npz"))):
        o, i = np.load(f), np.load(f.replace("out_", "in_"))
        c, lab, g = o["video team + dead"].astype(np.int8), i["lab"], i["g"]
        l3 = merged(lab, 3.0)
        n = os.path.basename(f)[4:-4]
        rows[n] = {k: score(lab, l3, shift_onsets(c, g, k)) for k in KS}
        names.append(n)
        print(n, {k: round(v[0], 4) for k, v in rows[n].items()}, flush=True)
    table(rows, names)


def st():
    from mtah import semimarkov as sm
    rows, names = {}, []
    for f in sorted(glob.glob(os.path.join(ROOT, "cache", "p9fc", "*.npz"))):
        if "grid" in f:
            continue
        z = np.load(f)
        g = z["g"]
        c = np.where(g, sm.decode_halves(z["p"], z["dead"].astype(np.float64), g, z["b"].astype(np.float64)), pm.X).astype(np.int8)
        n = os.path.basename(f)[:-4]
        rows[n] = {k: score(z["lab"], z["l3"], shift_onsets(c, g, k)) for k in KS}
        names.append(n)
        print(n, {k: round(v[0], 4) for k, v in rows[n].items()}, flush=True)
    table(rows, names)


if __name__ == "__main__":
    {"p9di": p9di, "st": st}[sys.argv[1]]()
