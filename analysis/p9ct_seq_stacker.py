"""P9ct: a small sequence-model stacker over v19's out-of-fold members (rule registered in notes/BULGULAR.md, 12 Nov,
before the run). Per frame: the six members' log-odds, the boundary probability, the two dead-ball probabilities, the
members' mean and spread -> a bidirectional GRU (48) -> P(home). Nested leave-one-match-out: the stacker of a match is
trained on the other nine matches' out-of-fold rows; team-swap augmentation; three seeds. The chain is v19's with the
stacker's probability in place of the members' mean. Local GPU.

    python analysis/p9ct_seq_stacker.py
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from m1_dead_classifier import to_blocks  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p9b_oof_sweeps import HI, LO  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

C = lambda *p: os.path.join(ROOT, "cache", *p)
L, EPOCHS, SEEDS, HID = 1500, 8, (0, 1, 2), 48
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))


def load():
    out = []
    for f in sorted(glob.glob(C("oof_tcn128d3", "*.npz"))):
        n = os.path.basename(f)
        z, big, t1, d3 = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), f))
        mem = np.stack([lg(x) for x in (big["pg"], d3["pt"], z["ps"], big["mg"], t1["mt"], z["ms"])], 1).astype(np.float32)
        mem = np.nan_to_num(mem)
        other = np.stack([np.nan_to_num(z["b"]), np.nan_to_num(z["qh"]), np.nan_to_num(z["qg"])], 1).astype(np.float32)
        out.append({"name": n[4:-4], "mem": mem, "other": other, "g": z["g"], "lab": z["labels"], "z": z,
                    "p": (big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6})
    return out


def feats(mem, other, sign=1.0):
    m = sign * mem
    return np.concatenate([m / 4, m.mean(1, keepdims=True) / 4, mem.std(1, keepdims=True) / 4, other], 1)


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, HID, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * HID, 1)

    def forward(self, x):
        return self.out(self.gru(x)[0]).squeeze(-1)


def fit_predict(train, test, seed, dev):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = feats(train[0]["mem"], train[0]["other"]).shape[1]
    net = Net(d).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    clips = [(i, s) for i, m in enumerate(train) for s in range(0, len(m["g"]) - L, L // 2)]
    X = [(feats(m["mem"], m["other"]), feats(m["mem"], m["other"], -1.0)) for m in train]
    Y = [np.where(m["lab"] == pm.H, 1.0, np.where(m["lab"] == pm.A, 0.0, -1.0)).astype(np.float32) for m in train]
    steps = EPOCHS * (len(clips) // 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    for _ in range(EPOCHS):
        rng.shuffle(clips)
        for k in range(0, len(clips) - 31, 32):
            xb, yb = [], []
            for i, s in clips[k:k + 32]:
                fl = rng.random() < 0.5
                xb.append(X[i][1 if fl else 0][s:s + L])
                y = Y[i][s:s + L]
                yb.append(np.where(y >= 0, 1 - y, -1) if fl else y)
            x, y = torch.from_numpy(np.stack(xb)).to(dev), torch.from_numpy(np.stack(yb).astype(np.float32)).to(dev)
            mask = y >= 0
            loss = nn.functional.binary_cross_entropy_with_logits(net(x)[mask], y[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    def run(x, step=4000, ctx=500):                             # cuDNN refuses one 140k-frame sequence: chunks with context
        out = np.zeros(len(x), np.float32)
        with torch.no_grad():
            for s in range(0, len(x), step):
                lo, hi = max(s - ctx, 0), min(s + step + ctx, len(x))
                y = net(torch.from_numpy(np.ascontiguousarray(x[lo:hi]))[None].to(dev))[0].cpu().numpy()
                out[s:min(s + step, len(x))] = y[s - lo:s - lo + min(step, len(x) - s)]
        return out
    a, b = run(feats(test["mem"], test["other"])), run(feats(test["mem"], test["other"], -1.0))
    return 1 / (1 + np.exp(-(a.astype(np.float64) - b) / 2))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    M = load()
    rows = []
    for i, m in enumerate(M):
        train = M[:i] + M[i + 1:]
        ps = np.mean([fit_predict(train, m, s, dev) for s in SEEDS], axis=0)
        z, g, lab = m["z"], m["g"], m["lab"]
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & g
        sw = HI * (1 - z["b"]) + LO * z["b"]
        l3 = merged(lab, 3.0)
        sc = lambda p: np.array([pm.spell_scores(Lb, np.where(g, decode(p, g & ~dead, sw), pm.X).astype(np.int8), pm.valid_runs(Lb))["mean_f1"] for Lb in (lab, l3)])
        base, new, half = sc(m["p"]), sc(ps), sc((ps + m["p"]) / 2)
        alive = g & ~dead & np.isin(lab, (pm.H, pm.A))
        acc = lambda p: np.mean((p[alive] > 0.5) == (lab[alive] == pm.H))
        rows.append((m["name"], base, new - base, half - base))
        np.save(C("oof_stack_{}.npy".format(m["name"])), ps.astype(np.float32))
        print(m["name"], "base {:.4f} {:.4f} stacker {:+.4f} {:+.4f} mean with members {:+.4f} {:+.4f} | frame accuracy {:.4f} -> {:.4f}".format(
            *base, *(new - base), *(half - base), acc(m["p"]), acc(ps)), flush=True)
    for name, col in (("stacker", 2), ("mean of stacker and members (information)", 3)):
        d = np.array([r[col] for r in rows if "DFL" in r[0]])
        q = np.array([r[col] for r in rows if "DFL" not in r[0]])
        ok = d.mean(0)[0] > 0 and d.mean(0)[1] >= 0 and (q.mean(0) >= 0).all()
        print("{}: DFL {:+.4f} {:+.4f} ({} / 7); Metrica {:+.4f} {:+.4f} ({} / 3) -> {}".format(name, *d.mean(0), int((d[:, 0] > 0).sum()), *q.mean(0), int((q[:, 0] > 0).sum()),
                                                                                    "HOLDS" if ok else "does not hold"))


if __name__ == "__main__":
    main()
