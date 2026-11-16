"""P9er: Mali's component rebuilt by us (scripts/mali_components.py: his recipe, nets trained on Colab, booster rebuilt
here; external_rebuilt/) against his delivered one (external/) inside our v28-style chain on all ten held-out folds
(P9ea's judges: six production members out-of-fold + Mali's team at 0.25; dead = v27a's mix classifier + GRU + Mali's
P(dead); production boundary; semi-Markov decoder). The v29 build switched to the rebuilt file for reproducibility;
this checks the switch costs nothing on external data.

    /home/mehdi/venvs/mali/bin/python analysis/p9er_mali_rebuilt.py

Registered before the run (14 Nov): the rebuilt file is kept in v29 if it is not behind the delivered one by more than
0.005 on DFL or Metrica at either referee on the realistic judge (equivalence, not a gain test); otherwise v29 goes
back to the delivered file.
"""
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

FOLDS = tuple("DFL-MAT-" + m for m in ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")) + ("metrica1", "metrica2", "metrica3")


def main():
    rows = {}
    for name in FOLDS:
        n = "oof_{}.npz".format(name)
        z, big, t1, d3 = (dict(np.load(C(x, n), allow_pickle=True)) for x in ("oof_v16", "oof_big", "oof_tcn128", "oof_tcn128d3"))
        e = dict(np.load(C("p9ea", n)))
        g, lab = z["g"], z["labels"]
        l3 = merged(lab, 3.0)
        p6 = np.nan_to_num((big["pg"] + d3["pt"] + z["ps"] + big["mg"] + t1["mt"] + z["ms"]) / 6, nan=0.5)
        b = np.nan_to_num(z["b"])
        r = {}
        for arm, sub in (("delivered", ""), ("rebuilt", os.path.join("..", "external_rebuilt"))):
            M = theirs("mehmet_ali_topkara", name, g, sub)
            team = sg(0.75 * lg(p6) + 0.25 * lg(M["team"]))
            for jn, dead in (("old", (e["p_old_mix"] + e["q_old"] + M["p_dead"]) / 3), ("real", (e["p_v6_mix"] + e["q_v6"] + M["p_dead"]) / 3)):
                c = np.where(g, sm.decode_halves(team, dead.astype(np.float64), g, b), pm.X).astype(np.int8)
                r[(arm, jn)] = [pm.spell_scores(L, c, pm.valid_runs(L))["mean_f1"] for L in (lab, l3)]
        rows[name] = r
        print(name, {"{}/{}".format(*k): np.round(v, 4).tolist() for k, v in r.items()}, flush=True)
    worst = 0.0
    for jn in ("old", "real"):
        out = ["judge {}: rebuilt vs delivered".format(jn)]
        for src in ("DFL", "metrica"):
            ms = [m for m in rows if src in m]
            d = np.array([rows[m][("rebuilt", jn)] for m in ms]) - np.array([rows[m][("delivered", jn)] for m in ms])
            out.append("{} {:+.4f} / {:+.4f} ({}/{} ahead at 1.6 s)".format(src, *d.mean(0), int((d[:, 0] > 0).sum()), len(ms)))
            if jn == "real":
                worst = min(worst, d.mean(0).min())
        print(" | ".join(out))
    print("rule (realistic, no source/referee behind by more than 0.005): worst {:+.4f} -> {}".format(worst, "keep rebuilt" if worst >= -0.005 else "back to delivered"))


if __name__ == "__main__":
    main()
