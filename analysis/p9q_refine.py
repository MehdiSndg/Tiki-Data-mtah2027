"""A probability-only refinement stage (research report item 13; MS-TCN++: stages that see only the previous stage's
probabilities over the whole sequence beat feeding them features): per frame, at 5 fps, the inputs are P9a's
out-of-fold v14 team probability (logit), each member family's logit, the boundary GRU and multi-task boundary
heads, the dead-ball probability and the member spread; a small dilated TCN over the whole match predicts the team,
trained on the other out-of-fold matches (leave one match out, 3 seeds), its output averaged with v14's (alpha) and
decoded as production. Full chain. GPU, minutes.

    python analysis/p9q_refine.py <oof_dir>      # -> eda_out/reports/P9q_refine.md
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
from p9b_oof_sweeps import HI, LO, load  # noqa: E402

STEP, SEEDS, EPOCHS = 5, (3, 4, 5), 40
ALPHAS = (0.0, 0.5, 1.0)
OUT = os.path.join(ROOT, "eda_out", "reports", "P9q_refine.md")
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def inputs(z):
    single, multi = (z["pg"] + z["pt"] + z["ps"]) / 3, (z["mg"] + z["mt"] + z["ms"]) / 3
    v14 = (single + multi) / 2
    mem = np.stack([z[k] for k in ("pg", "pt", "ps", "mg", "mt", "ms")], 1)
    q = (z["qh"] + z["qg"]) / 2
    X = np.stack([lg(v14), lg(single), lg(multi), z["b"], (z["mbg"] + z["mbt"] + z["mbs"]) / 3, q, mem.std(1), z["g"].astype(float)], 1)
    return X.astype(np.float32), v14


def net(nf):
    import torch

    class TCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Conv1d(nf, 64, 1)
            self.blocks = torch.nn.ModuleList([torch.nn.Conv1d(64, 64, 3, padding=d, dilation=d) for d in (1, 2, 4, 8, 16, 32, 64)])
            self.out = torch.nn.Conv1d(64, 1, 1)

        def forward(self, x):                    # (B, nf, L)
            h = self.inp(x)
            for c in self.blocks:
                h = h + torch.relu(c(h))
            return self.out(h)[:, 0]

    return TCN()


def train(tr, seed, dev):
    import torch
    torch.manual_seed(seed)
    m = net(tr[0][0].shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        for X, y, w in tr:
            x = torch.from_numpy(X.T[None]).to(dev)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(m(x)[0], torch.from_numpy(y).to(dev), reduction="none")
                    * torch.from_numpy(w).to(dev)).sum() / max(1.0, float(w.sum()))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval()


def main(d):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Z = load(d)
    names = sorted(Z)
    data = {}
    for n in names:
        z = Z[n]
        X, v14 = inputs(z)
        y = (z["labels"] == pm.H).astype(np.float32)
        w = np.isin(z["labels"], (pm.H, pm.A)).astype(np.float32)
        data[n] = (X[::STEP], y[::STEP], w[::STEP], X, v14)
    rows = []
    for n in names:
        tr = [data[k][:3] for k in names if k != n]
        mu = np.concatenate([t[0] for t in tr]).mean(0)
        sd = np.concatenate([t[0] for t in tr]).std(0) + 1e-6
        tr = [((X - mu) / sd, y, w) for X, y, w in tr]
        z = Z[n]
        Xf = (data[n][3] - mu) / sd
        outs = []
        for s in SEEDS:
            mdl = train(tr, s, dev)
            with torch.no_grad():
                o = torch.sigmoid(mdl(torch.from_numpy(Xf[::STEP].T[None].astype(np.float32)).to(dev))[0]).cpu().numpy()
            outs.append(np.repeat(o, STEP)[:len(Xf)])
        pr = np.mean(outs, axis=0)
        dead = to_blocks((z["qh"] + z["qg"]) / 2) & z["g"]
        cost = HI * (1 - z["b"]) + LO * z["b"]
        r = {"match": n}
        for a in ALPHAS:
            p = (1 - a) * data[n][4] + a * pr
            r["alpha {}".format(a)] = pm.spell_scores(z["labels"], np.where(z["g"], decode(p, z["g"] & ~dead, cost), pm.X).astype(np.int8), pm.valid_runs(z["labels"]))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["alpha {}".format(a) for a in ALPHAS]
    lines = ["# P9q: yalnız olasılık gören rafine katman (OOF, tam zincir, LOMO, 3 tohum)", "", "`analysis/p9q_refine.py`.", "",
             "alpha 0 = v14; 1 = yalnız rafine katman.", "", "| maç | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        lines.append("| ort. {} | ".format(src) + " | ".join("{:.3f}".format(q[c].mean()) for c in cols) + " |")
    lines.append("")
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        for c in cols[1:]:
            lines.append("{} {}: v14'e fark {:+.3f}, {} / {}".format(src, c, (q[c] - q["alpha 0.0"]).mean(), int((q[c] > q["alpha 0.0"]).sum()), len(q)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
