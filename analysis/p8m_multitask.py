"""One GRU for the team call and the boundary together (two heads on a shared BiGRU) against v11-style separate
GRUs. DFL LOMO, team call (dead ball from the labels), fresh seeds (3, 4, 5). GPU.

    python analysis/p8m_multitask.py      # -> eda_out/reports/P8m_multitask.md

Training data as v11 (DFL + Metrica 1-3, swapped); team loss on alive frames with short possessions x2, boundary
loss on alive frames, summed. Decoded with the per-frame switch cost from the boundary head (SWITCH_HI, SWITCH_LO).
"""

import glob
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
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

SEEDS = (3, 4, 5)
OUT = os.path.join(ROOT, "eda_out", "reports", "P8m_multitask.md")


def mt_net(nf):
    import torch

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(nf, 128)
            self.gru = torch.nn.GRU(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = torch.nn.Linear(256, 2)

        def forward(self, x):
            h, _ = self.gru(torch.relu(self.inp(x)))
            return self.out(h)

    return Net()


def mt_train(data, seed, dev):
    """data: (F, y_team, w_team, y_bnd, w_bnd, g); G.train's windows and recipe with two summed BCE losses."""
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)
    wins = []
    for F, yt, wt, yb, wb, g in data:
        Y = np.stack([yt, yb], 1)
        Wt = np.stack([wt, wb], 1)
        for f, y, w in G._windows(F, Y, Wt, g):
            wins.append((f, y, w))
    net = mt_net(data[0][0].shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=G.LR)
    for _ in range(G.EPOCHS):
        order = rng.permutation(len(wins))
        for a in range(0, len(order), G.BATCH):
            batch = [wins[i] for i in order[a:a + G.BATCH]]
            L = max(len(b[0]) for b in batch)
            X = np.zeros((len(batch), L, batch[0][0].shape[1]), np.float32)
            Y = np.zeros((len(batch), L, 2), np.float32)
            W = np.zeros((len(batch), L, 2), np.float32)
            for i, (f, y, w) in enumerate(batch):
                X[i, :len(f)], Y[i, :len(f)], W[i, :len(f)] = f, y, w
            X, Y, W = (torch.from_numpy(v).to(dev) for v in (X, Y, W))
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(X), Y, reduction="none") * W).sum() / W.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


def mt_predict(net, F, g, dev):
    import torch
    p = np.full((len(F), 2), 0.5)
    e = np.diff(np.r_[0, g.astype(np.int8), 0])
    with torch.no_grad():
        for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            for s in range(a, b, G.CHUNK):
                lo, hi = max(a, s - G.STRIDE), min(b, s + G.CHUNK + G.STRIDE)
                q = torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray(F[lo:hi][None])).to(dev))).cpu().numpy()[0]
                z = min(b, s + G.CHUNK)
                p[s:z] = q[s - lo:z - lo]
    return p


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    ms += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    sw = [swap_teams(m) for m in ms]
    T, Ts = [features(m) for m in ms], [features(m) for m in sw]
    S = [G.short_mask(m["labels"]) for m in ms]
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    print("features done", flush=True)
    rows = []
    for t in range(len(ids)):
        ks = [k for k in range(len(ms)) if k != t]
        pairs = [(mm, F, k) for k in ks for mm, F in ((ms[k], T[k]), (sw[k], Ts[k]))]
        tt = [(F, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(S[k], 2.0, 1.0).astype(np.float32), mm["half"] > 0) for mm, F, k in pairs]
        tb = [(F, G.boundary_target(mm["labels"]), aw(mm), mm["half"] > 0) for mm, F, k in pairs]
        m = ms[t]
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        p_sep, b_sep = fit_predict(tt, (T[t], g), seeds=SEEDS), fit_predict(tb, (T[t], g), seeds=SEEDS)
        allF = np.concatenate([F for _, F, _ in pairs])
        mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
        md = [(mod.norm(F), a[1], a[2], b[1], b[2], a[3]) for (_, F, _), a, b in zip(pairs, tt, tb)]
        q = np.mean([mt_predict(mt_train(md, s, "cuda"), mod.norm(T[t]), g, "cuda") for s in SEEDS], axis=0)
        sc = lambda p, b: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, G.SWITCH_HI * (1 - b) + G.SWITCH_LO * b), pm.X).astype(np.int8),
                                          pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"], "separate": sc(p_sep, b_sep), "multitask": sc(q[:, 0], q[:, 1]), "mt_team_sep_bnd": sc(q[:, 0], b_sep)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["separate", "multitask", "mt_team_sep_bnd"]
    lines = ["# P8m: takım ve sınır tek ağda (çok görevli GRU; DFL, ölü top etiketten, tohum 3–5)", "", "`analysis/p8m_multitask.py`.", "",
             "| maç | ayrı GRU'lar | çok görevli | çok görevli takım + ayrı sınır |", "|---|---|---|---|"]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", ""]
    lines += ["{}: ayrıya fark {:+.3f}, {} / 7 maçta önde.".format(c, (df[c] - df.separate).mean(), int((df[c] > df.separate).sum())) for c in cols[1:]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
