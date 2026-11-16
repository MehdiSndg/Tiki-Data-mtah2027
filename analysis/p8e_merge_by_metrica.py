"""DFL's label merge (merge_short) chosen with Metrica as the judge: Metrica's derived labels sit nearer the
public GT's possession counts than DFL's (notes/BULGULAR.md §1), and a DFL test cannot judge a change of
DFL's own target. Team call only, flat switch, fresh seeds (3, 4, 5). GPU.

    python analysis/p8e_merge_by_metrica.py [MERGE ...]      # -> eda_out/reports/P8e_merge_by_metrica[_<merges>].md

Per left-out Metrica game: the v10 team GRU (swapped, short x2) trained on the 7 DFL matches with their
labels merged at MERGES seconds (1.6 is production) + the other two Metrica games; scored on the Metrica game.
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
MERGES = (0.8, 1.6, 2.4)
OUT = os.path.join(ROOT, "eda_out", "reports", "P8e_merge_by_metrica.md")


def main(*merges):
    global MERGES, OUT
    if merges:
        MERGES = tuple(float(x) for x in merges)
        OUT = OUT.replace(".md", "_" + "_".join(merges) + ".md")
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    met = [degrade(from_metrica_sample(g), dict(rates_for("metrica"), smooth_frames=25), seed=0) for g in (1, 2, 3)]
    Tm, Tms = [features(m) for m in met], [features(swap_teams(m)) for m in met]
    dfl = {}
    for ms_ in MERGES:
        ms = [degrade(from_dfl(i, merge_s=ms_), dict(rates_for(i), smooth_frames=25), seed=0) for i in ids]
        dfl[ms_] = (ms, [features(m) for m in ms], [features(swap_teams(m)) for m in ms])
        print("merge", ms_, "spells/match", np.mean([len(pm.possession_spells(m["labels"], pm.valid_runs(m["labels"]))) for m in ms]), flush=True)
    def items(ms, T, Ts):
        out = []
        for m, F, Fs in zip(ms, T, Ts):
            s = swap_teams(m)
            sh = G.short_mask(m["labels"])
            for mm, FF in ((m, F), (s, Fs)):
                w = np.isin(mm["labels"], (pm.H, pm.A)).astype(np.float32) * np.where(sh, 2.0, 1.0).astype(np.float32)
                out.append((FF, (mm["labels"] == pm.H).astype(np.float32), w, mm["half"] > 0))
        return out
    rows = []
    for t in range(3):
        m = met[t]
        g = m["half"] > 0
        alive = np.isin(m["labels"], (pm.H, pm.A))
        sc = lambda q: pm.spell_scores(m["labels"], np.where(g, decode(q, alive, G.SWITCH), pm.X).astype(np.int8), pm.valid_runs(m["labels"]))["mean_f1"]
        others = [k for k in range(3) if k != t]
        r = {"match": m["name"]}
        for ms_ in MERGES:
            ms, T, Ts = dfl[ms_]
            data = items(ms, T, Ts) + items([met[k] for k in others], [Tm[k] for k in others], [Tms[k] for k in others])
            r["merge %g" % ms_] = sc(fit_predict(data, (Tm[t], g), seeds=SEEDS))
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    cols = ["merge %g" % x for x in MERGES]
    lines = ["# P8e: DFL etiket birleştirmesi, Metrica hakemliğinde (sadece takım, tohum 3–5)", "", "`analysis/p8e_merge_by_metrica.py`, birleştirmeler {}. Üretim 1.6 s.".format(MERGES), "",
             "| Metrica maçı | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += ["| {} | ".format(r["match"]) + " | ".join("{:.3f}".format(r[c]) for c in cols) + " |" for r in rows]
    lines += ["| ortalama | " + " | ".join("{:.3f}".format(df[c].mean()) for c in cols) + " |"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
