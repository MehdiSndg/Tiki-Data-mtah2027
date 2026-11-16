"""Fill the ATD ball's gaps from a second source (broadcast ATD ball, then ball_v1's tracked detections) before
the team GRU. DFL LOMO, team call only, 3 seeds. GPU.

    python analysis/p7b_ball_fill.py      # -> eda_out/reports/P7b_ball_fill.md

The P3 question left open (notes/BULGULAR.md §6: external data said fill, Smart Tagging said not) asked
again for the GRU. On our match (measured GT-free, 26 Oct) either source covers 59.6% of the in-play ATD
gaps; against the ATD ball they sit at median 0.85 m, 66% within 1.5 m, 12% beyond 5 m, in short stretches.
DFL is degraded as production and given a simulated second source at those rates (mtah.degrade.add_detector
with FILL_RATES); its position fills the degraded ball where that is missing. The GRU is trained and tested
without the fill, and with it (both sides filled). Dead ball from the labels.
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
from mtah.degrade import add_detector, degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl  # noqa: E402
from mtah.teammodel import decode, features  # noqa: E402
from p6v_gru_dead import gru_fit_predict  # noqa: E402

FILL_RATES = {"report": 0.6, "good": 0.75, "noise_m": 0.6, "chunk_s": 0.5, "wrong_near_player": 0.3}
OUT = os.path.join(ROOT, "eda_out", "reports", "P7b_ball_fill.md")


def filled(clean, deg, seed=0):
    d = add_detector(clean, deg, seed=seed, rates=FILL_RATES)
    miss = np.isnan(deg["bx"]) & ~np.isnan(d["dx"])
    out = dict(deg)
    out["bx"], out["by"] = deg["bx"].copy(), deg["by"].copy()
    out["bx"][miss], out["by"][miss] = d["dx"][miss], d["dy"][miss]
    out.pop("dx", None), out.pop("dy", None)
    return out, miss


def item(m, T):
    lab = m["labels"]
    return (T, (lab == pm.H).astype(np.float32), np.isin(lab, (pm.H, pm.A)).astype(np.float32), m["half"] > 0)


def score(m, p):
    alive = np.isin(m["labels"], (pm.H, pm.A))
    c = np.where(m["half"] > 0, decode(p, alive, G.SWITCH), pm.X).astype(np.int8)
    return pm.spell_scores(m["labels"], c, pm.valid_runs(m["labels"]))["mean_f1"]


def main():
    ids = sorted(q.rsplit("_", 1)[-1][:-4] for q in glob.glob(os.path.join(DFL, "*positions*.xml")))
    clean = [from_dfl(i) for i in ids]
    deg = [degrade(c, dict(rates_for(i), smooth_frames=25), seed=0) for c, i in zip(clean, ids)]
    fil = [filled(c, d) for c, d in zip(clean, deg)]
    for (f, miss), d in zip(fil, deg):
        gap = (d["half"] > 0) & np.isnan(d["bx"]) & np.isin(d["labels"], (pm.H, pm.A))
        print(d["name"], "in-play gaps filled %.1f%%" % (100 * miss[gap].mean()), flush=True)
    fil = [f for f, _ in fil]
    Td, Tf = [features(m) for m in deg], [features(m) for m in fil]
    rows = []
    for t in range(len(ids)):
        tr = [k for k in range(len(ids)) if k != t]
        r = {"match": deg[t]["name"],
             "base": score(deg[t], gru_fit_predict([item(deg[k], Td[k]) for k in tr], (Td[t], deg[t]["half"] > 0), seeds=G.SEEDS)),
             "fill": score(fil[t], gru_fit_predict([item(fil[k], Tf[k]) for k in tr], (Tf[t], fil[t]["half"] > 0), seeds=G.SEEDS))}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    d = df.fill - df.base
    lines = ["# P7b: ATD topunun boşluklarını ikinci kaynakla doldurmak (DFL, maç dışarıda bırakarak, 3 tohum, sadece takım)", "",
             "`analysis/p7b_ball_fill.py`, FILL_RATES = {}.".format(FILL_RATES), "",
             "| maç | doldurmadan | doldurarak |", "|---|---|---|"]
    lines += ["| {} | {:.3f} | {:.3f} |".format(r["match"], r["base"], r["fill"]) for r in rows]
    lines += ["| ortalama | {:.3f} | {:.3f} |".format(df.base.mean(), df.fill.mean()), "",
              "Fark {:+.3f} ({:+.3f} .. {:+.3f}); {} / {} maçta doldurmak önde.".format(d.mean(), d.min(), d.max(), int((d > 0).sum()), len(d))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
