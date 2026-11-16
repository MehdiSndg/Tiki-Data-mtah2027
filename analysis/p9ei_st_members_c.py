"""P9ei: recipe C's six team members (mtah.members_c, trained on Colab) against production's six on SoccerTrack's full
chain, a judge independent of our simulation (real tracking, degraded as P9bw; SoccerTrack never seen by either).
Everything but the team members is P9eg's v26 row: the 0.5 x smoothed dense video evidence added in log-odds, P(dead)
= P9di's dead3 (v8 classifier, GRU, video head), production's boundary GRU, the semi-Markov decoder. The evidence is
recovered from P9di's stored inputs as logit(team) - logit(team0) and checked to rebuild `team`.

    .venv/bin/python analysis/p9ei_st_members_c.py

Registered before the run (13 Nov, second judge for recipe C after P9ej): C must not be behind A at either referee
(1.6 s / 3 s, mean of 8 matches). Arm A must reproduce P9eg's v26 0.7235 / 0.7216. Inference only: a missing model
stops the run.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import grumodel as G  # noqa: E402
from mtah import semimarkov as sm  # noqa: E402
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import from_soccertrack  # noqa: E402
from mtah.members_c import external_gru_c  # noqa: E402
from p9bw_st_chain_base import ARCHES, DRAWS, no_training  # noqa: E402
from p9cs_st_video_dead import TEST  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402

logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))


def main():
    G.train = no_training
    members = [external_gru_c(arch=a, multitask=mt, seeds=tuple(range(6)), draws=DRAWS.get((a, mt), 1)) for a in ARCHES for mt in (False, True)]
    rows = {}
    for mid in [m for k in ("A", "B") for m in TEST[k]]:
        z = dict(np.load(os.path.join(ROOT, "cache", "p9di", "in_{}.npz".format(mid))))
        m0 = from_soccertrack(mid, merge_s=1.6)
        m = degrade(m0, dict(rates_for(m0["name"]), smooth_frames=25, dead_carry_p=0.25), seed=0)
        g, lab, b, dead = z["g"], z["lab"], z["b"].astype(np.float64), z["dead3"].astype(np.float64)
        assert len(g) == len(m["bx"])
        e = logit(z["team"].astype(np.float64)) - logit(z["team0"].astype(np.float64))
        err = np.abs(1 / (1 + np.exp(-(logit(z["team0"].astype(np.float64)) + e))) - z["team"])[g].max()
        pc = np.mean([G.predict(n, m) for n in members], axis=0)
        teams = {"A": z["team"].astype(np.float64), "C": 1 / (1 + np.exp(-(logit(pc) + e)))}
        l3 = merged(lab, 3.0)
        r = {}
        for k, t in teams.items():
            codes = np.where(g, sm.decode_halves(t, dead, g, b), pm.X).astype(np.int8)
            r[k] = [pm.spell_scores(L, codes, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        alive = g & np.isin(lab, (pm.H, pm.A))
        acc = {k: np.mean((teams[k][alive] > 0.5) == (lab[alive] == pm.H)) for k in teams}
        rows[mid] = r
        print(mid, "evidence rebuild err {:.1e}".format(err), {k: np.round(v, 4).tolist() for k, v in r.items()},
              "team acc A {:.3f} C {:.3f}".format(acc["A"], acc["C"]), flush=True)
    a, c = np.array([rows[m]["A"] for m in rows]), np.array([rows[m]["C"] for m in rows])
    print("SoccerTrack 8 | chain 1.6 s / 3 s")
    print("A (production) {:.4f} / {:.4f}".format(*a.mean(0)))
    print("C (recipe C)   {:.4f} / {:.4f}".format(*c.mean(0)))
    d = (c - a).mean(0)
    print("diff {:+.4f} / {:+.4f}; C ahead at 1.6 s in {}/8 -> {}".format(*d, int(((c - a)[:, 0] > 0).sum()), "holds" if (d >= 0).all() else "FAILS"))


if __name__ == "__main__":
    main()
