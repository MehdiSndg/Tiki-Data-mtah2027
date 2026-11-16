"""Metrica games 1-3 in the team GRU's training, with the team swap. Tested both ways, team call only. GPU.

    python analysis/p7i_metrica_gru.py      # -> eda_out/reports/P7i_metrica_gru.md

The two label proxies differ (DFL: BallPossession merged at 1.6 s; Metrica: our derivation from its
events, nearer the public GT's counts, notes/BULGULAR.md §1), so a gain on one only could be the other's
definition leaking in. DFL test: trained on the other six DFL, with or without the three Metrica games.
Metrica test: trained on the seven DFL, with or without the other two Metrica games. Every training match
also team-swapped; degraded as production (smooth 25, seed 0); flat switch; dead ball from the labels.
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

OUT = os.path.join(ROOT, "eda_out", "reports", "P7i_metrica_gru.md")


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    dfl = [degrade(from_dfl(i), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
    met = [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    ms = dfl + met
    sw = [swap_teams(m) for m in ms]
    T, Ts = [features(m) for m in ms], [features(m) for m in sw]
    print("features done", flush=True)
    it = lambda m, F: (F, (m["labels"] == pm.H).astype(np.float32), np.isin(m["labels"], (pm.H, pm.A)).astype(np.float32), m["half"] > 0)
    tr_set = lambda ks: [it(ms[k], T[k]) for k in ks] + [it(sw[k], Ts[k]) for k in ks]
    nd = len(dfl)
    rows = []
    for t in range(len(ms)):
        m = ms[t]
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        sc = lambda p: pm.spell_scores(m["labels"], np.where(g, decode(p, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        dk = [k for k in range(nd) if k != t]
        mk = [k for k in range(nd, len(ms)) if k != t]
        r = {"match": m["name"], "test": "DFL" if t < nd else "Metrica",
             "without": sc(fit_predict(tr_set(dk), (T[t], g))), "with": sc(fit_predict(tr_set(dk + mk), (T[t], g)))}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    lines = ["# P7i: Metrica 1–3 takım GRU'sunun eğitiminde (takım değişimiyle, sadece takım kararı, 3 tohum)", "", "`analysis/p7i_metrica_gru.py`.", "",
             "| maç | Metrica'sız | Metrica'lı |", "|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} |".format(r["match"], r["without"], r["with"]) for r in rows]
    for test in ("DFL", "Metrica"):
        d = df[df.test == test]
        lines += ["", "{} testi: {:.3f} → {:.3f} ({:+.3f}, {} / {} maçta önde).".format(test, d["without"].mean(), d["with"].mean(), (d["with"] - d["without"]).mean(),
                                                                                    int((d["with"] > d["without"]).sum()), len(d))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
