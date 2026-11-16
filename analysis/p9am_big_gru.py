"""A bigger team GRU with today's recipe (the teammates' "BigGRU"; P7k tried 256 hidden / 3 layers before Metrica,
multi-task members and 6 seeds, and found +0.001 / +0.003): the team GRU alone, production size (128 hidden, 2 layers)
vs 256 hidden and 3 layers, each averaged over 6 seeds (3-8), DFL leave one match out and Metrica 1-3, team call only,
flat switch; also scored against labels merged at 3.0 s. GPU.

    python analysis/p9am_big_gru.py [only=<match>,...]      # -> eda_out/reports/P9am_big_gru.md
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
from p9l_rescore_calibrated import merged  # noqa: E402


SEEDS = (3, 4, 5, 6, 7, 8)
FPS = 25
ONLY = [x for a in sys.argv[1:] if a.startswith("only=") for x in a[5:].split(",") if x]
VARIANTS = ("gru", "big")
OUT = os.path.join(ROOT, "eda_out", "reports", "P9am_big_gru{}.md".format("_" + "_".join(ONLY) if ONLY else ""))


def big_net(nf):
    import torch

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(nf, 256)
            self.gru = torch.nn.GRU(256, 256, num_layers=3, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = torch.nn.Linear(512, 1)

        def forward(self, x):
            h, _ = self.gru(torch.relu(self.inp(x)))
            return self.out(h).squeeze(-1)

    return Net()


def fit_predict(train, test, make_net=None, jitter=False):
    allF = np.concatenate([d[0] for d in train])
    mod = G.GRUEnsemble([], np.nanmean(allF, axis=0), np.nanstd(allF, axis=0) + 1e-6, "cuda")
    data = [(mod.norm(F), y, w, g) for F, y, w, g in train]
    nets = [G.train(data, s, "cuda", make_net=make_net, jitter=jitter) for s in SEEDS]
    return np.mean([G._predict_one(n, mod.norm(test[0]), test[1], "cuda") for n in nets], axis=0)


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    raws = [(from_dfl(i), i) for i in ids] + [(from_metrica_sample(g), "metrica") for g in (1, 2, 3)]
    base = [degrade(r, dict(rates_for(k), smooth_frames=25), seed=0) for r, k in raws]
    feats = [(m, features(m), features(swap_teams(m))) for m in base]
    print("features done", flush=True)
    aw = lambda m: np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32)
    rows = []
    names = [m["name"] for m in base]
    for t in range(len(raws)):
        if ONLY and not any(o in names[t] for o in ONLY):
            continue
        r = {"match": names[t]}
        tr = []
        for k, (m, F, Fs) in enumerate(feats):
            if k == t:
                continue
            sw, sh = swap_teams(m), G.short_mask(m["labels"])
            for mm, FF in ((m, F), (sw, Fs)):
                tr.append((FF, (mm["labels"] == pm.H).astype(np.float32), aw(mm) * np.where(sh, 2.0, 1.0).astype(np.float32), mm["half"] > 0))
        m, F, _ = feats[t]
        g, alive = m["half"] > 0, np.isin(m["labels"], (pm.H, pm.A))
        for v, p in (("gru", fit_predict(tr, (F, g))), ("big", fit_predict(tr, (F, g), make_net=big_net))):
            pred = np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8)
            r[v] = pm.spell_scores(m["labels"], pred, pm.valid_runs(m["labels"]))["mean_f1"]
            l3 = merged(m["labels"], 3.0)
            r[v + " @3s"] = pm.spell_scores(l3, pred, pm.valid_runs(l3))["mean_f1"]
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P9am: büyük takım GRU'su (256 gizli, 3 katman) vs üretim (6 tohum, sadece takım)", "", "`analysis/p9am_big_gru.py`.", "",
             "| maç | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[v]) for v in VARIANTS) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[v].mean()) for v in VARIANTS) + " |", ""]
    for src in ("DFL", "metrica"):
        q = df[df.match.str.startswith(src)]
        if len(q):
            for sfx in ("", " @3s"):
                lines.append("{}{}: big − gru {:+.3f}, {} / {}".format(src, sfx, (q["big" + sfx] - q["gru" + sfx]).mean(), int((q["big" + sfx] > q["gru" + sfx]).sum()), len(q)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
