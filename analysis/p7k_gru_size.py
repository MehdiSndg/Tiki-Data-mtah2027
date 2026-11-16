"""GRU size and training length again, now that the team swap doubles the data. DFL LOMO, team call only,
flat switch, 3 seeds. GPU.

    python analysis/p7k_gru_size.py      # -> eda_out/reports/P7k_gru_size.md

All with the team swap (P7e). swap: production (128 hidden, 2 layers, 8 epochs); ep12: 12 epochs; h256:
256 hidden; l3: 3 layers.
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

OUT = os.path.join(ROOT, "eda_out", "reports", "P7k_gru_size.md")


def net(nf, hidden=128, layers=2):
    import torch

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(nf, hidden)
            self.gru = torch.nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = torch.nn.Linear(2 * hidden, 1)

        def forward(self, x):
            h, _ = self.gru(torch.relu(self.inp(x)))
            return self.out(h).squeeze(-1)

    return Net()


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
        r = {"match": m["name"], "swap": sc(fit_predict(data, (T[t], g)))}
        G.EPOCHS = 12
        r["ep12"] = sc(fit_predict(data, (T[t], g)))
        G.EPOCHS = 8
        r["h256"] = sc(fit_predict(data, (T[t], g), make_net=lambda nf: net(nf, hidden=256)))
        r["l3"] = sc(fit_predict(data, (T[t], g), make_net=lambda nf: net(nf, layers=3)))
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["swap", "ep12", "h256", "l3"]
    lines = ["# P7k: GRU boyu ve eğitim süresi, takım değişimiyle (DFL, maç dışarıda bırakarak, sadece takım, 3 tohum)", "", "`analysis/p7k_gru_size.py`.", "",
             "| maç | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |", ""]
    lines += ["{}: swap'a fark {:+.3f}, {} / 7 maçta önde.".format(c, (df[c] - df.swap).mean(), int((df[c] > df.swap).sum())) for c in cols[1:]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
