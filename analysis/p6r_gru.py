"""A bidirectional GRU over the team features, against and averaged with the booster. LOMO on DFL. GPU.

    python analysis/p6r_gru.py      # -> eda_out/reports/P6r_gru.md

Same data as P6b's touch_motion variant (degraded DFL, ball smoothed over 25 frames, no extra copies):
mtah.teammodel features with touches and motion, standardised on the training matches, NaN -> 0.
Model: linear 128 -> 2-layer BiGRU 128 -> P(home), BCE on alive frames, 30 s windows (stride 15 s),
8 epochs, Adam 1e-3; inference in 60 s chunks with 15 s of context each side. Per test match (trained on the other six): spellF1
of the GRU, the booster, and their mean probability, each through the Viterbi at the production switch.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.teammodel import SWITCH, decode, fit, team_features  # noqa: E402

WIN, STRIDE, EPOCHS, BATCH, SEED = 750, 375, 8, 32, 0
OUT = os.path.join(ROOT, "eda_out", "reports", "P6r_gru.md")


def windows(F, y, w, g):
    out = []
    e = np.diff(np.r_[0, g.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        for s in range(a, max(a + 1, b - WIN + 1), STRIDE):
            z = min(b, s + WIN)
            out.append((F[s:z], y[s:z], w[s:z]))
    return out


def train_gru(train, nf, dev):
    import torch
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(nf, 128)
            self.gru = torch.nn.GRU(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = torch.nn.Linear(256, 1)

        def forward(self, x):
            h, _ = self.gru(torch.relu(self.inp(x)))
            return self.out(h).squeeze(-1)

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        order = rng.permutation(len(train))
        for a in range(0, len(order), BATCH):
            batch = [train[i] for i in order[a:a + BATCH]]
            L = max(len(b[0]) for b in batch)
            X = np.zeros((len(batch), L, nf), np.float32)
            Y = np.zeros((len(batch), L), np.float32)
            Wt = np.zeros((len(batch), L), np.float32)
            for i, (f, y, w) in enumerate(batch):
                X[i, :len(f)], Y[i, :len(f)], Wt[i, :len(f)] = f, y, w
            X, Y, Wt = (torch.from_numpy(v).to(dev) for v in (X, Y, Wt))
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(X), Y, reduction="none") * Wt).sum() / Wt.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print("  epoch", ep, round(loss.item(), 4), flush=True)
    return net


def predict(net, F, g, dev):
    import torch
    p = np.full(len(F), 0.5)
    e = np.diff(np.r_[0, g.astype(np.int8), 0])
    net.eval()
    with torch.no_grad():
        for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            for s in range(a, b, 1500):                     # 60 s chunks with 15 s of context each side
                lo, hi = max(a, s - STRIDE), min(b, s + 1500 + STRIDE)
                x = torch.from_numpy(np.ascontiguousarray(F[lo:hi][None])).to(dev)
                q = torch.sigmoid(net(x)).cpu().numpy()[0]
                z = min(b, s + 1500)
                p[s:z] = q[s - lo:z - lo]
    return p


def spell(m, c):
    return pm.spell_scores(m["labels"], c, pm.valid_runs(m["labels"]))["mean_f1"]


def main():
    dev = "cuda"
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    D = []
    for m in ms:
        F, _ = team_features(m, True, True)
        D.append((F, np.isin(m["labels"], (pm.H, pm.A)), (m["labels"] == pm.H).astype(np.float32), m["half"] > 0))
        print(m["name"], flush=True)
    rows = []
    for t in range(len(ms)):
        tr = [k for k in range(len(ms)) if k != t]
        allF = np.concatenate([D[k][0] for k in tr])
        mu, sd = np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6
        norm = lambda F: np.nan_to_num((F - mu) / sd).astype(np.float32).clip(-10, 10)
        train = [w for k in tr for w in windows(norm(D[k][0]), D[k][2], D[k][1].astype(np.float32), D[k][3])]
        net = train_gru(train, allF.shape[1], dev)
        m = ms[t]
        F, alive, _, g = D[t]
        pg = predict(net, norm(F), g, dev)
        pb = fit([D[k][0][D[k][1]] for k in tr], [D[k][2][D[k][1]].astype(int) for k in tr]).predict_proba(F)[:, 1]
        r = {"match": m["name"], "gru": spell(m, decode(pg, alive, SWITCH)), "booster": spell(m, decode(pb, alive, SWITCH)),
             "mean": spell(m, decode((pg + pb) / 2, alive, SWITCH))}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P6r: iki yönlü GRU, booster ve ortalamaları (DFL, maç dışarıda bırakarak)", "", "`analysis/p6r_gru.py`. Sadece takım kararı.", "",
             "| maç | booster | GRU | ortalama |", "|---|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} | {:.3f} |".format(r["match"], r["booster"], r["gru"], r["mean"]) for r in rows]
    lines += ["", "Ortalama: booster {:.3f}, GRU {:.3f}, ortalama {:.3f}; ortalama − booster {:+.3f} ({} / {} maçta önde).".format(
        df.booster.mean(), df.gru.mean(), df["mean"].mean(), (df["mean"] - df.booster).mean(), int((df["mean"] > df.booster).sum()), len(df))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
