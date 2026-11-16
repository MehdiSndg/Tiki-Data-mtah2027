"""GRU variants, two seeds each, team call only, LOMO on DFL (switch cost 8). GPU.

    python analysis/p6t_gru_variants.py      # -> eda_out/reports/P6t_gru_variants.md

base     as P6r/P6s (mtah.grumodel defaults: 30 s windows, 8 epochs, hidden 128)
copies   training also on mtah.teammodel.COPIES (unsmoothed ball, degradation seeds 1-2)
wide     hidden 256
long     60 s windows (stride 30 s), 12 epochs
The mean probability of the two seeds is decoded and scored per test match.
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
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.teammodel import COPIES, decode, features  # noqa: E402

SEEDS = (0, 1)
OUT = os.path.join(ROOT, "eda_out", "reports", "P6t_gru_variants.md")


def set_hidden(h):
    import torch

    def net(nf):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = torch.nn.Linear(nf, h)
                self.gru = torch.nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
                self.out = torch.nn.Linear(2 * h, 1)

            def forward(self, x):
                y, _ = self.gru(torch.relu(self.inp(x)))
                return self.out(y).squeeze(-1)
        return Net()
    G._net = net


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [from_dfl(i) for i in ids]
    ms = [degrade(r, dict(rates_for(i), smooth_frames=25), seed=0) for r, i in zip(raws, ids)]
    base = [(features(m), m) for m in ms]
    extra = [[(features(e), e) for e in (degrade(r, dict(rates_for(i), smooth_frames=sm), seed=sd) for sm, sd in COPIES)]
             for r, i in zip(raws, ids)]
    print("features done", flush=True)
    orig_net, orig = G._net, (G.WIN, G.STRIDE, G.EPOCHS)
    res = {}
    for variant in ("base", "copies", "wide", "long"):
        G._net = orig_net
        G.WIN, G.STRIDE, G.EPOCHS = orig
        if variant == "wide":
            set_hidden(256)
        if variant == "long":
            G.WIN, G.STRIDE, G.EPOCHS = 60 * 25, 30 * 25, 12
        scores = []
        for t in range(len(ms)):
            tr = [k for k in range(len(ms)) if k != t]
            pairs = [base[k] for k in tr] + ([x for k in tr for x in extra[k]] if variant == "copies" else [])
            allF = np.concatenate([F for F, _ in pairs])
            mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
            data = [(mod.norm(F), (m["labels"] == pm.H).astype(np.float32), np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32), m["half"] > 0)
                    for F, m in pairs]
            mod.nets = [G.train(data, s, "cuda") for s in SEEDS]
            F, m = base[t]
            p = np.mean([G._predict_one(n, mod.norm(F), m["half"] > 0, "cuda") for n in mod.nets], axis=0)
            c = decode(p, np.isin(m["labels"], (pm.H, pm.A)), G.SWITCH)
            scores.append(pm.spell_scores(m["labels"], c, pm.valid_runs(m["labels"]))["mean_f1"])
            print(variant, m["name"], round(scores[-1], 3), flush=True)
        res[variant] = scores
    df = pd.DataFrame(res, index=[m["name"] for m in ms])
    lines = ["# P6t: GRU varyantları (DFL, maç dışarıda bırakarak, 2 tohum ortalaması)", "", "`analysis/p6t_gru_variants.py`. Sadece takım kararı, ceza 8.", "",
             "| maç | " + " | ".join(df.columns) + " |", "|---|" + "---|" * len(df.columns)]
    lines += ["| {} | ".format(i) + " | ".join("{:.3f}".format(v) for v in r) + " |" for i, r in df.iterrows()]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(v) for v in df.mean()) + " |"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
