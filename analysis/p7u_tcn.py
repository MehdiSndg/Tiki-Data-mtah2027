"""A temporal CNN beside the team GRU: alone and averaged with it. DFL LOMO, team call only, flat switch,
3 seeds each, all with the team swap. GPU.

    python analysis/p7u_tcn.py      # -> eda_out/reports/P7u_tcn.md

TCN: linear 128 -> six residual blocks of non-causal dilated Conv1d (kernel 5, dilation 1..32, ~10 s of
context each side) -> logit; trained with mtah.grumodel.train's recipe (it only needs net(x) -> logits).
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
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.match import swap_teams  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

OUT = os.path.join(ROOT, "eda_out", "reports", "P7u_tcn.md")


def tcn(nf, hidden=128, dilations=(1, 2, 4, 8, 16, 32), k=5):
    import torch

    class Block(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.c1 = torch.nn.Conv1d(hidden, hidden, k, padding=d * (k - 1) // 2, dilation=d)
            self.c2 = torch.nn.Conv1d(hidden, hidden, 1)
            self.drop = torch.nn.Dropout(0.1)

        def forward(self, x):
            return x + self.c2(self.drop(torch.relu(self.c1(x))))

    class TCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(nf, hidden)
            self.blocks = torch.nn.Sequential(*[Block(d) for d in dilations])
            self.out = torch.nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.blocks(torch.relu(self.inp(x)).transpose(1, 2)).transpose(1, 2)
            return self.out(torch.relu(h)).squeeze(-1)

    return TCN()


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    sw = [swap_teams(m) for m in ms]
    T, Ts = [features(m) for m in ms], [features(m) for m in sw]
    print("features done", flush=True)
    it = lambda m, F: (F, (m["labels"] == pm.H).astype(np.float32), np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32), m["half"] > 0)
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        data = [it(ms[k], T[k]) for k in tr] + [it(sw[k], Ts[k]) for k in tr]
        m = ms[t]
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        sc = lambda p: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        pg = fit_predict(data, (T[t], g))
        pt = fit_predict(data, (T[t], g), make_net=tcn)
        r = {"match": m["name"], "gru": sc(pg), "tcn": sc(pt), "mean": sc((pg + pt) / 2)}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["gru", "tcn", "mean"]
    lines = ["# P7u: zamansal CNN, GRU'nun yanında (DFL, maç dışarıda bırakarak, sadece takım, takım değişimiyle, 3 tohum)", "", "`analysis/p7u_tcn.py`.", "",
             "| maç | GRU | TCN | ortalama |", "|---|---|---|---|"]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", ""]
    lines += ["{}: GRU'ya fark {:+.3f}, {} / 7 maçta önde.".format(c, (df[c] - df.gru).mean(), int((df[c] > df.gru).sum())) for c in cols[1:]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
