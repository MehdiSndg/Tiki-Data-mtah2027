"""P9el: where the remaining loss is, and how much the ruler moves the number. v28's chain on held-out DFL 7 + Metrica 3
(v26's fused team, P9ea's held-out dead ball, semi-Markov decoder) on the old (production) and the realistic copies:
  * oracle dead: the labels' D (team from the chain)       -> the dead ball's share of the loss
  * oracle team: the labels' team on live frames (dead from the chain) -> the team call's share
  * ruler: the same predictions scored against labels merged at 0 / 1.6 / 3 / 5 s (our referees are 1.6 and 3 s)
"Oracle" numbers are ceilings ON THESE COPIES only (feedback-validate-the-judge).

    /home/mehdi/venvs/mali/bin/python analysis/p9el_decompose.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from mtah import semimarkov as sm  # noqa: E402
from p9de_team_fuse import C, lg, sg, theirs  # noqa: E402
from p9l_rescore_calibrated import merged  # noqa: E402


def main():
    rows = {}
    for f in sorted(glob.glob(C("p9ea", "oof_*.npz"))):
        n = os.path.basename(f)
        name = n[4:-4]
        z, big, t1, d3, e = (dict(np.load(x, allow_pickle=True)) for x in (C("oof_v16", n), C("oof_big", n), C("oof_tcn128", n), C("oof_tcn128d3", n), f))
        g, lab = z["g"], z["labels"]
        b = np.nan_to_num(z["b"])
        p = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
        M = theirs("mehmet_ali_topkara", name, g)
        team = sg(0.75 * lg(p) + 0.25 * lg(M["team"]))
        truth_team = np.where(lab == pm.H, 0.999, np.where(lab == pm.A, 0.001, team))
        truth_dead = np.where(lab == pm.D, 0.999, 0.001)
        refs = {k: (merged(lab, k) if k else lab) for k in (0.0, 1.6, 3.0, 5.0)}
        sc = lambda c: {k: pm.spell_scores(L, np.where(g, c, pm.X).astype(np.int8), pm.valid_runs(L))["mean_f1"] for k, L in refs.items()}
        r = {}
        for judge, dead in (("old", (e["p_old_prod"] + e["q_old"] + M["p_dead"]) / 3), ("real", (e["p_v6_mix"] + e["q_v6"] + M["p_dead"]) / 3)):
            dead = dead.astype(np.float64)
            r[(judge, "chain")] = sc(sm.decode_halves(team, dead, g, b))
            r[(judge, "oracle dead")] = sc(sm.decode_halves(team, truth_dead, g, b))
            r[(judge, "oracle team")] = sc(sm.decode_halves(truth_team, dead, g, b))
        rows[name] = r
        print(name, "done", flush=True)
    for src in ("DFL", "metrica"):
        ms = [m for m in rows if src in m]
        print("==", src, len(ms), "matches | referees: labels merged 0 / 1.6 / 3 / 5 s")
        for key in rows[ms[0]]:
            v = np.array([[rows[m][key][k] for k in (0.0, 1.6, 3.0, 5.0)] for m in ms]).mean(0)
            print("  {:<5} {:<12} {:.4f} / {:.4f} / {:.4f} / {:.4f}".format(*key, *v))


if __name__ == "__main__":
    main()
