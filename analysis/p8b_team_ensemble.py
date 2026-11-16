"""A wider team ensemble on v10's data: GRU and TCN (v10) plus a Transformer and the player-set GRU
(mtah.setgru). DFL LOMO, team call only, flat switch, fresh seeds (3, 4, 5). GPU.

    python analysis/p8b_team_ensemble.py      # -> eda_out/reports/P8b_team_ensemble.md

All members trained on DFL + Metrica 1-3, every match also team-swapped, short possessions x2 (v10). The
GRU + TCN mean gave v10 its largest gain; more different members may add to it.
"""

import glob
import itertools
import math
import os
import sys
from functools import partial

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
from mtah.setgru import player_set, set_net  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p7d_player_set import fit_predict  # noqa: E402

SEEDS = (3, 4, 5)
OUT = os.path.join(ROOT, "eda_out", "reports", "P8b_team_ensemble.md")


def transformer(nf, d=128, layers=3, heads=4):
    import torch

    class TRF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(nf, d)
            enc = torch.nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True)
            self.enc = torch.nn.TransformerEncoder(enc, layers)
            self.out = torch.nn.Linear(d, 1)

        def forward(self, x):
            T = x.shape[1]
            pos = torch.arange(T, device=x.device).float()[:, None]
            div = torch.exp(torch.arange(0, d, 2, device=x.device).float() * (-math.log(10000.0) / d))
            pe = torch.zeros(T, d, device=x.device)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            return self.out(self.enc(torch.relu(self.inp(x)) + pe)).squeeze(-1)

    return TRF()


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    ms = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    ms += [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    sw = [swap_teams(m) for m in ms]
    T, Ts = [features(m) for m in ms], [features(m) for m in sw]
    P = [np.concatenate([t, player_set(m)], 1) for t, m in zip(T, ms)]
    Ps = [np.concatenate([t, player_set(m)], 1) for t, m in zip(Ts, sw)]
    S = [G.short_mask(m["labels"]) for m in ms]
    nd, nft = len(ids), T[0].shape[1]
    print("features done", flush=True)
    def data(ks, A, As):
        out = []
        for k in ks:
            for m, F in ((ms[k], A[k]), (sw[k], As[k])):
                w = np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32) * np.where(S[k], 2.0, 1.0).astype(np.float32)
                out.append((F, (m["labels"] == pm.H).astype(np.float32), w, m["half"] > 0))
        return out
    members = ("gru", "tcn", "set", "trf")
    rows = []
    for t in range(nd):
        ks = [k for k in range(len(ms)) if k != t]
        m = ms[t]
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        p = {"gru": fit_predict(data(ks, T, Ts), (T[t], g), seeds=SEEDS),
             "tcn": fit_predict(data(ks, T, Ts), (T[t], g), make_net=G._tcn, seeds=SEEDS),
             "set": fit_predict(data(ks, P, Ps), (P[t], g), make_net=partial(lambda nf, n0: set_net(n0), n0=nft), seeds=SEEDS),
             "trf": fit_predict(data(ks, T, Ts), (T[t], g), make_net=transformer, seeds=SEEDS)}
        sc = lambda q: pm.spell_scores(m["labels"], np.where(g, decode(q, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        r = {"match": m["name"]}
        for n in range(1, len(members) + 1):
            for combo in itertools.combinations(members, n):
                r["+".join(combo)] = sc(np.mean([p[c] for c in combo], axis=0))
        rows.append(r)
        print({k: round(v, 3) for k, v in r.items() if k != "match"}, flush=True)
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c != "match"]
    ref = df["gru+tcn"]
    lines = ["# P8b: daha geniş takım topluluğu (v10 verisi, DFL, maç dışarıda bırakarak, sadece takım, tohum 3–5)", "", "`analysis/p8b_team_ensemble.py`.", "",
             "| üyeler | ortalama | v10'a (gru+tcn) fark | önde |", "|---|---|---|---|"]
    for c in sorted(cols, key=lambda c: -df[c].mean()):
        lines.append("| {} | {:.3f} | {:+.3f} | {} / 7 |".format(c, df[c].mean(), (df[c] - ref).mean(), int((df[c] > ref).sum())))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
