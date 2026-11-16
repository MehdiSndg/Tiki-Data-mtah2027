"""Choose the control rule's R and T on degraded DFL, check them on degraded Metrica. GT-free for us.

    uv run python analysis/m3_control_params.py

The dead ball is given from the labels, so this measures the team call alone. R and T were fixed
at 1.5 m / 0.4 s from clean DFL and Metrica (analysis/g4_dfl_rules.py); on our ATD's kind of data
(degraded) the best pair may differ: the ATD loses players near the ball and duplicates some.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from mtah.degrade import degrade, rates_for  # noqa: E402
from mtah.external import DFL, from_dfl, from_metrica_sample  # noqa: E402
from mtah.harness import evaluate  # noqa: E402
from mtah.labelers import control  # noqa: E402

RS = (1.0, 1.5, 2.0, 3.0)
TS = (0.2, 0.4, 0.6, 0.8, 1.2)


def main():
    ids = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
    dfl = [degrade(from_dfl(i), rates_for(i), seed=0) for i in ids]
    met = [degrade(from_metrica_sample(g), rates_for("metrica"), seed=0) for g in (1, 2)]
    res = {}
    print("spellF1, dead ball from labels   (DFL degraded = selection | Metrica degraded = check)")
    print("   R \\ T  " + "".join("{:>15}".format(t) for t in TS))
    for R in RS:
        row = []
        for T in TS:
            d = evaluate(lambda m: control(m, R=R, T=T, dead="truth"), dfl, bootstrap=0).spell_f1.mean()
            k = evaluate(lambda m: control(m, R=R, T=T, dead="truth"), met, bootstrap=0).spell_f1.mean()
            res[(R, T)] = (d, k)
            row.append("{:.3f} | {:.3f}".format(d, k))
        print("   {:<6} ".format(R) + "".join("{:>15}".format(x) for x in row))
    best = max(res, key=lambda k: res[k][0])
    print("chosen on DFL: R={} T={} -> DFL {:.3f}, Metrica {:.3f} (current 1.5 / 0.4: DFL {:.3f}, Metrica {:.3f})".format(
        *best, *res[best], *res[(1.5, 0.4)]))


if __name__ == "__main__":
    main()
