"""A learned switch cost (PathCRF's lesson: an input-dependent transition score fitted with the chain's likelihood sharpens
boundaries): on v17's out-of-fold chain (v16 OOF with P9at's BigGRU members), the per-frame cost of a change of team is
c_t = softplus(theta . phi_t), phi_t = [1, b_t, logit b_t, near_dead_t] (b: the boundary GRU's probability; near_dead:
exp(-frames to the nearest predicted dead block / 25)), and the emissions are scaled by exp(a). (theta, a) maximise the
two-state linear-chain CRF log-likelihood of the 1.6 s training labels on the true alive runs of the other nine matches
(nested leave one match out, DFL and Metrica together); the held-out match is decoded with the production Viterbi using
c_t / exp(a), dead ball as in the OOF, and scored at 1.6 s and against labels merged at 3.0 s, against v17's cost
12 (1 - b) + 0.5 b. CPU (torch).

    python analysis/p9be_crf_switch.py      # -> eda_out/reports/P9be_crf_switch.md
"""

import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p9b_oof_sweeps import HI, LO  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

C = lambda *p: os.path.join(ROOT, "cache", *p)
OUT = os.path.join(ROOT, "eda_out", "reports", "P9be_crf_switch.md")
CHUNK = 2000
FIT_SCALE = False
NEG = -1e9


def near_dead(dead):
    """exp(-distance in frames to the nearest dead frame / 25)."""
    n = len(dead)
    idx = np.flatnonzero(dead)
    if not len(idx):
        return np.zeros(n)
    pos = np.searchsorted(idx, np.arange(n))
    d = np.minimum(np.abs(np.arange(n) - idx[np.clip(pos - 1, 0, len(idx) - 1)]), np.abs(idx[np.clip(pos, 0, len(idx) - 1)] - np.arange(n)))
    return np.exp(-d / 25.0)


def load():
    out = []
    for f in sorted(glob.glob(C("oof_v16", "*.npz"))):
        n = os.path.basename(f)
        z, big = dict(np.load(f, allow_pickle=True)), dict(np.load(C("oof_big", n), allow_pickle=True))
        g, lab = z["g"], z["labels"]
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        p = (big["pg"] + z["pt"] + z["ps"] + big["mg"] + z["mt"] + z["ms"]) / 6
        b = np.clip(z["b"].astype(np.float64), 1e-4, 1 - 1e-4)
        phi = np.stack([np.ones_like(b), b, np.log(b / (1 - b)), near_dead(dead)], 1)
        out.append(dict(name=n[4:-4], g=g, lab=lab, dead=dead, p=p, b=b, phi=phi))
    return out


def crf_terms(m):
    """The training sequence of one match: frames whose label is H or A, with run starts where the alive run breaks."""
    keep = np.isin(m["lab"], (pm.H, pm.A))
    idx = np.flatnonzero(keep)
    start = np.r_[True, np.diff(idx) > 1]
    p = np.clip(m["p"][idx], 1e-4, 1 - 1e-4)
    y = (m["lab"][idx] == pm.A).astype(np.int64)                       # 0 = H, 1 = A
    return dict(lh=np.log(p), la=np.log1p(-p), y=y, start=start, phi=m["phi"][idx])


def loglik(theta, a, T):
    """Mean per-frame CRF log-likelihood over the concatenated sequences in T (list of crf_terms)."""
    lh = torch.tensor(np.concatenate([t["lh"] for t in T]))
    la = torch.tensor(np.concatenate([t["la"] for t in T]))
    y = torch.tensor(np.concatenate([t["y"] for t in T]))
    start = torch.tensor(np.concatenate([t["start"] for t in T]))
    phi = torch.tensor(np.concatenate([t["phi"] for t in T]))
    s = torch.exp(a)
    c = torch.nn.functional.softplus(phi @ theta)
    c = torch.where(start, torch.zeros_like(c), c)                    # a free change where a run starts: runs independent
    em = torch.stack([s * lh, s * la], 1)                              # (n, 2)
    n = len(y)
    gold = em[torch.arange(n), y].sum() - (c[1:] * (y[1:] != y[:-1]).double()).sum()
    # log-partition via chunked log-semiring products of the 2x2 step matrices M_t[i, j] = trans_t(i -> j) + em_t(j)
    pad = (-n) % CHUNK
    trans = torch.stack([torch.stack([torch.zeros_like(c), -c], 1), torch.stack([-c, torch.zeros_like(c)], 1)], 1)   # (n, 2, 2)
    M = trans + em[:, None, :]
    eye = torch.full((pad, 2, 2), NEG, dtype=M.dtype)
    eye[:, 0, 0] = eye[:, 1, 1] = 0.0
    M = torch.cat([M, eye]).view(-1, CHUNK, 2, 2)
    A = M[:, 0]
    for t in range(1, CHUNK):
        A = torch.logsumexp(A[:, :, :, None] + M[:, t][:, None, :, :], dim=2)
    tot = A[0]
    for k in range(1, A.shape[0]):
        tot = torch.logsumexp(tot[:, :, None] + A[k][None, :, :], dim=1)
    # the first frame's own emission enters through M_0's columns; the row (previous state) is free: logsumexp over it
    logZ = torch.logsumexp(tot, dim=(0, 1)) - np.log(2.0)
    return (gold - logZ) / n


def fit(T, iters=40):
    theta = torch.tensor([np.log(np.expm1(HI)), -(HI - LO) / 2, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    # the emission scale stays at 1 (the production decode): fitted freely it fell to ~0.01 on the first fold (cost ~600,
    # one possession per run, spellF1 0.70 -> 0.41), because neighbouring frames' probabilities are not independent
    a = torch.zeros((), dtype=torch.float64, requires_grad=FIT_SCALE)
    opt = torch.optim.LBFGS([theta, a] if FIT_SCALE else [theta], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -loglik(theta, a, T)
        loss.backward()
        return loss
    opt.step(closure)
    return theta.detach().numpy(), float(a.detach())


def score(m, cost):
    c = np.where(m["g"], decode(m["p"], m["g"] & ~m["dead"], cost), pm.X).astype(np.int8)
    lab = m["lab"]
    l3 = merged(lab, 3.0)
    return pm.spell_scores(lab, c, pm.valid_runs(lab))["mean_f1"], pm.spell_scores(l3, c, pm.valid_runs(l3))["mean_f1"]


def main():
    torch.set_num_threads(4)
    ms = load()
    T = [crf_terms(m) for m in ms]
    rows = []
    for t, m in enumerate(ms):
        theta, a = fit([T[k] for k in range(len(ms)) if k != t])
        cost = np.log1p(np.exp(m["phi"] @ theta)) / np.exp(a)
        base = HI * (1 - m["b"]) + LO * m["b"]
        r = (m["name"],) + score(m, base) + score(m, cost) + (float(np.median(cost[m["g"]])), )
        rows.append(r)
        print(r, "theta", np.round(theta, 3), "a %.3f" % a, flush=True)
    lines = ["# P9be: öğrenen geçiş cezası (CRF olabilirliği, iç içe LOMO), v17 OOF zinciri", "", "`analysis/p9be_crf_switch.py`.", "",
             "| maç | v17 | v17 @3s | crf | crf @3s | medyan ceza |", "|---|---|---|---|---|---|"]
    lines += ["| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |".format(*r) for r in rows]
    lines.append("")
    d = {}
    for src in ("DFL", "metrica"):
        q = np.array([r[1:5] for r in rows if r[0].startswith(src)])
        dd = q[:, 2:4] - q[:, :2]
        d[src] = dd.mean(0)
        lines.append("{}: crf − v17 1.6 s {:+.4f} ({} / {}), 3 s {:+.4f} ({} / {})".format(src, dd[:, 0].mean(), int((dd[:, 0] > 0).sum()), len(dd), dd[:, 1].mean(), int((dd[:, 1] > 0).sum()), len(dd)))
    ok = d["DFL"][0] > 0 and d["DFL"][1] >= 0 and d["metrica"][0] >= 0 and d["metrica"][1] >= 0
    lines.append("karar: " + ("kural tuttu" if ok else "alınmadı"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
