"""P9cr: a Bayes decision for spellF1 on v19's out-of-fold chain (rule registered in notes/BULGULAR.md, 11 Nov, before
the run). Per alive run: paths ~ exp(beta * Viterbi score) by forward filtering, backward sampling (K paths); candidate
boundaries from the samples' switch mass and the Viterbi path; each candidate possession (i, j, team) is worth the share
of (sample, IoU threshold) pairs in which the sample has a possession of that team with at least that IoU; a dynamic
programme over alternating teams maximises the sum of (worth - lam). Nested leave-one-match-out over (beta, lam). CPU.

    python analysis/p9cr_mbr_decode.py
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
K, WIN, MASS, MAXC = 64, 6, 0.05, 40
BETAS, LAMS = (0.1, 0.2, 0.4, 0.8), (0.30, 0.37, 0.45)
IOUS = np.array([0.3, 0.4, 0.5, 0.6, 0.7])


def sample_paths(lh, la, c, beta, rng):
    """(K, n) states (0 home, 1 away) ~ exp(beta * (sum of log emissions - sum of switch costs))."""
    n = len(lh)
    eh, ea, tr = np.exp(beta * (lh - np.maximum(lh, la))), np.exp(beta * (la - np.maximum(lh, la))), np.exp(-beta * c)
    A = np.empty((n, 2))
    h, a = eh[0], ea[0]
    s = h + a
    A[0] = h / s, a / s
    for t in range(1, n):
        h, a = (A[t - 1, 0] + A[t - 1, 1] * tr[t]) * eh[t], (A[t - 1, 1] + A[t - 1, 0] * tr[t]) * ea[t]
        s = h + a
        A[t] = h / s, a / s
    S = np.empty((K, n), np.int8)
    S[:, n - 1] = rng.random(K) < A[n - 1, 1]
    U = rng.random((n, K))
    for t in range(n - 2, -1, -1):
        nxt = S[:, t + 1]
        w0 = A[t, 0] * np.where(nxt == 0, 1.0, tr[t + 1])
        w1 = A[t, 1] * np.where(nxt == 1, 1.0, tr[t + 1])
        S[:, t] = U[t] * (w0 + w1) >= w0
    return S


def mbr_run(S, vit, lam):
    n = S.shape[1]
    sw = np.concatenate([np.zeros((K, 1), bool), S[:, 1:] != S[:, :-1]], axis=1)
    mass = np.convolve(sw.mean(0), np.ones(2 * WIN + 1), mode="same")
    cand = set(np.flatnonzero(np.r_[False, vit[1:] != vit[:-1]]).tolist())
    order = np.argsort(-mass)
    picked = []
    for t in order[:400]:
        if mass[t] < MASS or len(picked) >= MAXC:
            break
        if all(abs(t - q) > WIN for q in picked):
            picked.append(int(t))
    cand = np.array(sorted(cand | set(picked) | {0, n}))
    cand = cand[(cand == 0) | (cand == n) | ((cand > 2) & (cand < n - 2))]
    # all sample spells: start, end, team, sample
    st, en, tm = [], [], []
    for k in range(K):
        e = np.flatnonzero(sw[k])
        a, b = np.r_[0, e], np.r_[e, n]
        st.append(a); en.append(b); tm.append(S[k, a])
    cnt = np.array([len(x) for x in st])
    sid = np.repeat(np.arange(K), cnt)
    st, en, tm = np.concatenate(st), np.concatenate(en), np.concatenate(tm)
    m = len(cand)
    best = np.full((m, 2), -np.inf)
    back = np.zeros((m, 2), np.int32)
    best[0] = 0.0
    for j in range(1, m):
        for i in range(max(0, j - 12), j):
            a, b = cand[i], cand[j]
            inter = np.clip(np.minimum(en, b) - np.maximum(st, a), 0, None)
            iou = inter / ((en - st) + (b - a) - inter)
            for s in (0, 1):
                prev = best[i, 1 - s] if i > 0 else 0.0
                if not np.isfinite(prev):
                    continue
                mx = np.zeros(K)
                np.maximum.at(mx, sid[tm == s], iou[tm == s])
                v = prev + np.mean(mx[:, None] >= IOUS[None, :]) - lam
                if v > best[j, s]:
                    best[j, s], back[j, s] = v, i
    out = np.empty(n, np.int8)
    j, s = m - 1, int(np.argmax(best[m - 1]))
    while j > 0:
        i = back[j, s]
        out[cand[i]:cand[j]] = s
        j, s = i, 1 - s
    return out


def main():
    rng = np.random.default_rng(0)
    R = {}
    names = []
    for f in sorted(glob.glob(C("oof_tcn128d3", "*.npz"))):
        n = os.path.basename(f)
        names.append(n[4:-4])
        z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f))
        g, lab = z["g"], z["labels"]
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        alive = g & ~dead
        l3 = merged(lab, 3.0)
        p = np.clip((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, 1e-4, 1 - 1e-4)
        sw = np.broadcast_to(HI * (1 - z["b"]) + LO * z["b"], p.shape)
        lh, la = np.log(p), np.log(1 - p)
        vit = decode(p, alive, sw)
        sc = lambda c: np.array([pm.spell_scores(lab, c, pm.valid_runs(lab))["mean_f1"], pm.spell_scores(l3, c, pm.valid_runs(l3))["mean_f1"]])
        base = sc(np.where(g, vit, pm.X).astype(np.int8))
        e = np.diff(np.r_[0, alive.astype(np.int8), 0])
        runs = list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))
        for beta in BETAS:
            samples = [sample_paths(lh[a:b], la[a:b], sw[a:b], beta, rng) if b - a > 5 else None for a, b in runs]
            for lam in LAMS:
                c = vit.copy()
                for (a, b), S in zip(runs, samples):
                    if S is None:
                        continue
                    o = mbr_run(S, (vit[a:b] == pm.A).astype(np.int8), lam)
                    c[a:b] = np.where(o == 0, pm.H, pm.A)
                R.setdefault((beta, lam), []).append(sc(np.where(g, c, pm.X).astype(np.int8)) - base)
            print(n[4:-4], "beta", beta, " ".join("{:+.4f}/{:+.4f}".format(*R[(beta, lam)][-1]) for lam in LAMS), flush=True)
    keys = list(R)
    G = np.array([R[k] for k in keys])                       # (settings, matches, 2)
    print("setting means (all matches):")
    for k, v in zip(keys, G):
        print(" beta {} lam {} : {:+.4f} {:+.4f}".format(*k, *v.mean(0)))
    nested = []
    for i in range(len(names)):
        others = np.delete(G, i, axis=1).mean(axis=(1, 2))
        nested.append(G[int(np.argmax(others)), i])
    nested = np.array(nested)
    d = np.array(["DFL" in x for x in names])
    print("nested: DFL {:+.4f} {:+.4f} ({} / {} ahead at 1.6 s); Metrica {:+.4f} {:+.4f} ({} / {})".format(
        *nested[d].mean(0), int((nested[d][:, 0] > 0).sum()), int(d.sum()), *nested[~d].mean(0), int((nested[~d][:, 0] > 0).sum()), int((~d).sum())))
    ok = nested[d].mean(0)[0] > 0 and nested[d].mean(0)[1] >= 0 and (nested[~d].mean(0) >= 0).all()
    print("rule:", "HOLDS" if ok else "does not hold")


if __name__ == "__main__":
    main()
