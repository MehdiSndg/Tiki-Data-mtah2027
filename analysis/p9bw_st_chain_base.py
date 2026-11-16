"""P9bw: the base of the video evidence's gate B (notes/BULGULAR.md, 9 Nov): v19c's team call on the SoccerTrack v2
matches, inference only (the six team members of v19c with six seeds each and its boundary GRU, all trained on DFL +
Metrica; SoccerTrack never seen). Tracking degraded at our match's rates (rates_for, ball smoothed over 25 frames,
dead_carry_p 0.25, seed 0), as the out-of-fold test matches. Rulings written before any result: the dead ball is taken
from the labels in both arms (the video evidence only touches the team call, so the difference is clean; the dead model
has never seen SoccerTrack); no restart evidence (no Smart Tagging there). Labels: the ball file's status, merged at
1.6 s (and 3 s for the second referee). Folds fixed before any result: A = 117092 117093 118575 118576 132831,
B = 118577 118578 128057 128058 132877 (132831 and 132877 have no possession status: crops only).

Saves cache/st_chain/<match>.npz (p, switch, labels, g) and prints spellF1 per match. GPU (inference); stops if a
model is missing instead of training it.

    python analysis/p9bw_st_chain_base.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import from_soccertrack, soccertrack_usable  # noqa: E402
from mtah.teammodel import decode  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

FOLDS = {"A": ("117092", "117093", "118575", "118576", "132831"), "B": ("118577", "118578", "128057", "128058", "132877")}
ARCHES = ("big", "tcn128", "set")                     # scripts/make_submission.py V18_ARCHES
DRAWS = {("tcn128", False): 3}                       # V19_DRAWS
OUT = os.path.join(ROOT, "cache", "st_chain")


def no_training(*a, **k):
    raise RuntimeError("a v19c model is missing from models/: refusing to train it here")


def main():
    G.train = no_training
    os.makedirs(OUT, exist_ok=True)
    members = [G.external_gru(swap=True, metrica=True, short_k=2.0, arch=a, multitask=mt, seeds=tuple(range(6)), draws=DRAWS.get((a, mt), 1))
               for a in ARCHES for mt in (False, True)]
    bnd = G.external_gru(target="boundary", swap=True, metrica=True)
    for fold, mids in FOLDS.items():
        for mid in mids:
            m0 = from_soccertrack(mid, merge_s=1.6)
            if not soccertrack_usable(m0):
                continue
            m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
            g = m["half"] > 0
            lab = m0["labels"]
            p = np.mean([G.predict(e, m) for e in members], axis=0)
            sw = G.switch_costs(bnd, m)
            np.savez_compressed(os.path.join(OUT, "{}.npz".format(mid)), p=p, switch=sw, labels=lab, g=g, fold=fold)
            alive = g & np.isin(lab, (pm.H, pm.A))
            c = np.where(g, np.where(alive, decode(p, alive, sw), pm.D), pm.X).astype(np.int8)
            l3 = merged(lab, 3.0)
            print(fold, mid, "{:.4f} {:.4f}".format(pm.spell_scores(lab, c, pm.valid_runs(lab))["mean_f1"],
                                                    pm.spell_scores(l3, c, pm.valid_runs(l3))["mean_f1"]), flush=True)


if __name__ == "__main__":
    main()
