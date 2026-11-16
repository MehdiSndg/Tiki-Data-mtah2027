"""P9ew: a third, independent judge for v30 (twelve seeds per recipe-C member) against v29 (six): PFF FC's 2022 World
Cup broadcast tracking (real tracking with real gaps; never seen by any member), the first 12 matches of
mtah.pff.pff_ids(), ball only where PFF saw it (ball="visible", the ATD-like case). Team call only, as P9bw on
SoccerTrack: the dead ball from the labels in both arms, production boundary GRU's switch costs, mtah.teammodel.decode.
PFF's labels merged at 1.6 s (and 3 s for the second referee).

    .venv/bin/python analysis/p9ew_pff_seeds.py

Registered before the run (15 Nov, after P9ev split 1-1: SoccerTrack for v30, video agreement against by 0.0003):
v30 is recommended to the user only if it is not behind v29 on PFF at either referee (mean of the 12 matches); the
override of P9eu's video rule stays the user's decision either way. Inference only: a missing model stops the run.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.pff import from_pff, pff_ids  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p9bw_st_chain_base import no_training  # noqa: E402
from p9ev_seeds12 import members, team  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402


def main():
    ms = members()
    bnd = G.external_gru(target="boundary", swap=True, metrica=True)
    G.train = no_training
    rows = {}
    for gid in pff_ids()[:12]:
        m = from_pff(gid, merge_s=1.6, ball="visible")
        g, lab = m["half"] > 0, m["labels"]
        l3 = merged(lab, 3.0)
        alive = g & np.isin(lab, (pm.H, pm.A))
        p6, p12 = team(ms, m)
        sw = G.switch_costs(bnd, m)
        r = {}
        for k, p in (("v29", p6), ("v30", p12)):
            c = np.where(g, np.where(alive, decode(p, alive, sw), pm.D), pm.X).astype(np.int8)
            r[k] = [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        acc = {k: np.mean((p[alive] > 0.5) == (lab[alive] == pm.H)) for k, p in (("v29", p6), ("v30", p12))}
        rows[gid] = r
        print(gid, {k: np.round(v, 4).tolist() for k, v in r.items()}, "team acc v29 {:.3f} v30 {:.3f}".format(acc["v29"], acc["v30"]), flush=True)
    a, c = np.array([rows[k]["v29"] for k in rows]), np.array([rows[k]["v30"] for k in rows])
    d = (c - a).mean(0)
    print("PFF {}: v29 {:.4f} / {:.4f}, v30 {:.4f} / {:.4f}, diff {:+.4f} / {:+.4f} ({}/{} ahead at 1.6 s) -> {}".format(
        len(rows), *a.mean(0), *c.mean(0), *d, int(((c - a)[:, 0] > 0).sum()), len(rows), "holds" if (d >= 0).all() else "FAILS"))


if __name__ == "__main__":
    main()
